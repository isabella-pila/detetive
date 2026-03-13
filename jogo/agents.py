"""
agents.py — Lógica dos agentes LLM do jogo Murdle.

Três agentes:
  Agent 1 - Narrador: gera a história, solução secreta e alibis
  Agent 2 - Suspeito: responde interrogatórios em personagem
  Agent 3 - Investigador: analisa tentativas e gera dicas contextuais

O EstadoJogo é serializado na sessão Django entre requests.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ==============================
# LLM
# ==============================

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ==============================
# ESTADO DO JOGO
# ==============================

class EstadoJogo:
    """
    Estado completo de uma partida.
    Serializado como dict para armazenar na sessão Django.
    """

    PERSONAGENS = ["mordomo", "herdeira", "jardineiro"]
    ARMAS       = ["veneno", "faca", "corda"]
    LOCAIS      = ["biblioteca", "cozinha", "jardim"]

    def __init__(self):
        self.historia: str = ""
        self.solucao: dict = {}
        self.alibis: dict = {}
        self.historico: list = []   # [{personagem, pergunta, resposta}]
        self.tentativas: int = 0
        self.max_tentativas: int = 5
        self.game_over: bool = False
        self.vitoria: bool = False

    # ── serialização ───────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "historia": self.historia,
            "solucao": self.solucao,
            "alibis": self.alibis,
            "historico": self.historico,
            "tentativas": self.tentativas,
            "max_tentativas": self.max_tentativas,
            "game_over": self.game_over,
            "vitoria": self.vitoria,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EstadoJogo":
        obj = cls()
        obj.historia       = data.get("historia", "")
        obj.solucao        = data.get("solucao", {})
        obj.alibis         = data.get("alibis", {})
        obj.historico      = data.get("historico", [])
        obj.tentativas     = data.get("tentativas", 0)
        obj.max_tentativas = data.get("max_tentativas", 5)
        obj.game_over      = data.get("game_over", False)
        obj.vitoria        = data.get("vitoria", False)
        return obj

    # ── helpers ────────────────────────────────────────────────────
    def registrar_interrogatorio(self, personagem: str, pergunta: str, resposta: str):
        self.historico.append({
            "personagem": personagem,
            "pergunta": pergunta,
            "resposta": resposta,
        })

    def resumo_historico(self) -> str:
        if not self.historico:
            return "Nenhum interrogatório realizado ainda."
        return "\n".join(
            f"[{e['personagem'].upper()}] P: {e['pergunta']} | R: {e['resposta']}"
            for e in self.historico
        )

    @property
    def tentativas_restantes(self) -> int:
        return self.max_tentativas - self.tentativas


# ==============================
# AGENTE 1 — Narrador / Criador da História
# ==============================

_story_prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é o narrador de um jogo de mistério de assassinato no estilo Murdle.
Gere um mistério curto e atmosférico. O crime ocorreu em uma mansão durante uma tempestade.

Personagens (escolha UM como culpado): mordomo, herdeira, jardineiro
Locais (escolha UM): biblioteca, cozinha, jardim
Armas (escolha UMA): veneno, faca, corda

Regras:
1. Escolha SECRETAMENTE culpado, arma e local — NÃO os revele na história.
2. Escreva 3 parágrafos que plantem pistas SUTIS mas não óbvias.
3. Crie um álibi incompleto para cada personagem.
4. Responda SOMENTE com JSON válido. Sem markdown. Sem texto extra.

Formato:
{{
  "historia": "3 parágrafos atmosféricos",
  "pessoa": "mordomo|herdeira|jardineiro",
  "arma": "veneno|faca|corda",
  "local": "biblioteca|cozinha|jardim",
  "alibis": {{
    "mordomo": "alibi curto",
    "herdeira": "alibi curto",
    "jardineiro": "alibi curto"
  }}
}}"""),
    ("human", "Gere o mistério agora."),
])


def criar_historia(state: EstadoJogo) -> None:
    """Chama o Agente 1 e popula o EstadoJogo com história + solução."""
    llm = get_llm()
    chain = _story_prompt | llm
    resposta = chain.invoke({})

    conteudo = resposta.content.strip()
    # Remove blocos ```json``` que modelos menores às vezes incluem
    if conteudo.startswith("```"):
        conteudo = conteudo.split("```")[1]
        if conteudo.startswith("json"):
            conteudo = conteudo[4:]

    data = json.loads(conteudo)

    state.historia = data["historia"]
    state.solucao  = {
        "pessoa": data["pessoa"],
        "arma":   data["arma"],
        "local":  data["local"],
    }
    state.alibis = data.get("alibis", {})


# ==============================
# AGENTE 2 — Suspeito
# ==============================

_suspect_prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é {personagem} em um jogo de detetive de assassinato.

História do crime:
{historia}

Seu álibi (só você sabe — use para se defender):
{alibi}

Você é o culpado? {e_culpado}

Histórico de interrogatórios desta sessão:
{historico}

Regras:
Se for culpado escolha uma estratégia:

1) negar diretamente
2) mudar de assunto
3) insinuar suspeita sobre outro
4) responder parcialmente
- Se NÃO for culpado: responda honestamente, mas pode ter pequenos segredos.
- Nunca quebre o personagem.
- 2 a 4 frases curtas, estilo noir dramático, primeira pessoa."""),
    ("human", "Pergunta do detetive: {pergunta}"),
])


def interrogar_suspeito(personagem: str, pergunta: str, state: EstadoJogo) -> str:
    """Chama o Agente 2 e registra a resposta no histórico."""
    llm = get_llm()
    chain = _suspect_prompt | llm

    e_culpado = "SIM" if personagem == state.solucao.get("pessoa") else "NÃO"
    alibi     = state.alibis.get(personagem, "Sem álibi registrado.")

    resposta = chain.invoke({
        "personagem": personagem,
        "historia":   state.historia,
        "alibi":      alibi,
        "e_culpado":  e_culpado,
        "historico":  state.resumo_historico(),
        "pergunta":   pergunta,
    })

    state.registrar_interrogatorio(personagem, pergunta, resposta.content)
    return resposta.content


# ==============================
# AGENTE 3 — Investigador / Gerador de Dicas
# ==============================

_hint_prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um investigador sênior guiando um detetive novato.
Conhece a solução real. Oriente sem revelar diretamente.

Regras:
1. NUNCA revele a resposta diretamente.
2. Se o jogador acertou algum campo, confirme isso.
3. Baseie a dica no histórico de interrogatórios quando possível.
4. Com 1 ou 2 tentativas restantes, seja mais específico (mas não entregue).
5. Seja dramático — veterano endurecido.
6. 2 a 3 frases."""),
    ("human", """Solução real:
- Culpado: {pessoa}
- Arma:    {arma}
- Local:   {local}

Tentativa do jogador:
- Culpado: {pessoa_user} ({resultado_pessoa})
- Arma:    {arma_user} ({resultado_arma})
- Local:   {local_user} ({resultado_local})

Tentativa {tentativa_num} de {max_tentativas} (restam {restantes}).

Histórico de interrogatórios:
{historico}

Gere a dica."""),
])


def gerar_dica(tentativa: dict, erros: list, acertos: list, state: EstadoJogo) -> str:
    """Chama o Agente 3 e retorna a dica contextual."""
    llm = get_llm()
    chain = _hint_prompt | llm

    resposta = chain.invoke({
        "pessoa":          state.solucao["pessoa"],
        "arma":            state.solucao["arma"],
        "local":           state.solucao["local"],
        "pessoa_user":     tentativa["pessoa"],
        "arma_user":       tentativa["arma"],
        "local_user":      tentativa["local"],
        "resultado_pessoa": "CORRETO" if "pessoa" in acertos else "ERRADO",
        "resultado_arma":   "CORRETO" if "arma"   in acertos else "ERRADO",
        "resultado_local":  "CORRETO" if "local"  in acertos else "ERRADO",
        "tentativa_num":   state.tentativas,
        "max_tentativas":  state.max_tentativas,
        "restantes":       state.tentativas_restantes,
        "historico":       state.resumo_historico(),
    })

    return resposta.content


# ==============================
# REGRA DE DECISÃO
# ==============================

def verificar_tentativa(solucao: dict, tentativa: dict) -> tuple[str, list, list]:
    """
    Compara tentativa com solução de forma normalizada.
    Retorna: (resultado, erros, acertos)
    """
    erros   = []
    acertos = []

    for campo in ["pessoa", "arma", "local"]:
        if tentativa[campo].strip().lower() == solucao[campo].strip().lower():
            acertos.append(campo)
        else:
            erros.append(campo)

    return ("acertou" if not erros else "errou"), erros, acertos
