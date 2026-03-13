import json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ==============================
# LLM
# ==============================

# CORREÇÃO: modelo correto é "gpt-4o-mini", não "gpt-5-mini"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ==============================
# ESTADO DO JOGO
# ==============================
# Centralizamos o estado aqui para que todos os agentes
# tenham acesso ao histórico completo de interrogatórios.
# Isso permite que o Agente 2 (suspeito) se lembre do que
# já foi dito, e o Agente 3 (dicas) use esse contexto.

class EstadoJogo:
    def __init__(self):
        self.historia: str = ""
        self.solucao: dict = {}
        self.alibis: dict = {}
        self.historico_interrogatorios: list = []
        self.tentativas: int = 0
        self.max_tentativas: int = 5

    def registrar_interrogatorio(self, personagem: str, pergunta: str, resposta: str):
        self.historico_interrogatorios.append({
            "personagem": personagem,
            "pergunta": pergunta,
            "resposta": resposta,
        })

    def resumo_historico(self) -> str:
        if not self.historico_interrogatorios:
            return "Nenhum interrogatorio realizado ainda."
        linhas = []
        for e in self.historico_interrogatorios:
            linhas.append(
                f"[{e['personagem'].upper()}] "
                f"Pergunta: {e['pergunta']} | "
                f"Resposta: {e['resposta']}"
            )
        return "\n".join(linhas)


# ==============================
# AGENTE 1 - Criador da Historia
# ==============================
# CORRECOES:
# - Modelo errado corrigido
# - Prompt pede alibis individuais para cada personagem
# - Instrucao explicita de nao revelar o culpado na historia
# - Parser JSON robusto que remove blocos ```json``` se necessario

story_prompt = ChatPromptTemplate.from_messages([
    ("system", """Voce e o narrador de um jogo de misterio de assassinato no estilo Murdle.
Gere um misterio curto e atmosferico. O crime ocorreu em uma mansao durante uma tempestade.

Personagens disponiveis (escolha UM como culpado):
- mordomo
- herdeira
- jardineiro

Locais disponiveis (escolha UM):
- biblioteca
- cozinha
- jardim

Armas disponiveis (escolha UMA):
- veneno
- faca
- corda

Regras:
1. Escolha SECRETAMENTE culpado, arma e local - NAO os revele na historia.
2. Escreva 3-4 paragrafos que plantem pistas SUTIS mas nao obvias.
3. Crie um alibi incompleto ou suspeito para cada personagem inocente.
4. Responda SOMENTE com JSON valido. Sem texto antes ou depois. Sem markdown.

Formato obrigatorio:
{{
  "historia": "3-4 paragrafos atmosfericos",
  "pessoa": "mordomo|herdeira|jardineiro",
  "arma": "veneno|faca|corda",
  "local": "biblioteca|cozinha|jardim",
  "alibis": {{
    "mordomo": "alibi breve do mordomo",
    "herdeira": "alibi breve da herdeira",
    "jardineiro": "alibi breve do jardineiro"
  }}
}}"""),
    ("human", "Gere o misterio agora."),
])

story_chain = story_prompt | llm


def criar_historia(state):
    resposta = story_chain.invoke({})

    # CORRECAO: parser mais robusto - remove blocos ```json``` se o modelo
    # os incluir por engano, algo comum em modelos menores.
    conteudo = resposta.content.strip()
    if conteudo.startswith("```"):
        conteudo = conteudo.split("```")[1]
        if conteudo.startswith("json"):
            conteudo = conteudo[4:]

    try:
        data = json.loads(conteudo)
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSON: {e}")
        print("Resposta bruta:\n", resposta.content)
        exit(1)

    state.historia = data["historia"]
    state.solucao = {
        "pessoa": data["pessoa"],
        "arma": data["arma"],
        "local": data["local"],
    }
    state.alibis = data.get("alibis", {})
    return data


# ==============================
# AGENTE 2 - Suspeitos
# ==============================
# CORRECOES:
# - Prompt original nao passava se o personagem era culpado ou nao
# - Agora passamos isso explicitamente para o LLM
# - Passamos o alibi individual de cada personagem
# - Passamos o historico completo de interrogatorios (memoria de sessao)
#   para que as respostas sejam coerentes entre si

suspect_prompt = ChatPromptTemplate.from_messages([
    ("system", """Voce e {personagem} em um jogo de detetive de assassinato.

Historia do crime:
{historia}

Seu alibi (apenas voce sabe - use-o para se defender se necessario):
{alibi}

Voce cometeu o crime? {e_culpado}

Historico desta investigacao (o que ja foi dito):
{historico}

Regras de comportamento:
- Se voce FOR culpado: defenda-se com conviccao. Desvie perguntas diretas com sutileza. Nunca confesse.
- Se voce NAO for culpado: responda honestamente, mas pode estar nervoso ou ter pequenos segredos nao relacionados ao crime.
- Nunca quebre o personagem.
- Responda em 2-4 frases curtas no estilo de um interrogatorio dramatico de noir.
- Fale sempre em primeira pessoa."""),
    ("human", "Pergunta do detetive: {pergunta}"),
])

suspect_chain = suspect_prompt | llm


def perguntar(personagem, pergunta, state):
    e_culpado = "SIM" if personagem == state.solucao["pessoa"] else "NAO"
    alibi = state.alibis.get(personagem, "Nao tenho alibi definido.")

    resposta = suspect_chain.invoke({
        "personagem": personagem,
        "historia": state.historia,
        "alibi": alibi,
        "e_culpado": e_culpado,
        "historico": state.resumo_historico(),
        "pergunta": pergunta,
    })

    # Registra no historico para contexto futuro
    state.registrar_interrogatorio(personagem, pergunta, resposta.content)
    return resposta.content


# ==============================
# AGENTE 3 - Gerador de Dicas
# ==============================
# CORRECOES:
# - Prompt original nao diferenciava campos certos dos errados
# - Nao tinha acesso ao historico, gerando dicas genericas
# - Agora o agente sabe o que o jogador acertou/errou
# - Escala a especificidade da dica conforme tentativas diminuem
# - Pode referenciar respostas dos interrogatorios

hint_prompt = ChatPromptTemplate.from_messages([
    ("system", """Voce e um investigador senior ajudando um detetive novato.
Voce conhece a solucao real e deve orienta-lo sem revelar diretamente.

Regras:
1. NAO revele a resposta diretamente em nenhuma hipotese.
2. Se o jogador acertou algum campo, confirme isso na dica.
3. Baseie a dica no historico de interrogatorios quando possivel.
4. Se restam poucas tentativas (1 ou 2), seja mais especifico, mas ainda sem entregar.
5. Seja dramatico - voce e um veterano endurecido.
6. Responda em 2-3 frases."""),
    ("human", """Solucao real:
- Culpado: {pessoa}
- Arma: {arma}
- Local: {local}

Tentativa do jogador:
- Culpado: {pessoa_user} ({resultado_pessoa})
- Arma: {arma_user} ({resultado_arma})
- Local: {local_user} ({resultado_local})

Tentativa {tentativa_num} de {max_tentativas} (restam {restantes}).

Historico de interrogatorios:
{historico}

Gere a dica."""),
])

hint_chain = hint_prompt | llm


def gerar_dica(state, tentativa, erros, acertos):
    restantes = state.max_tentativas - state.tentativas

    resposta = hint_chain.invoke({
        "pessoa": state.solucao["pessoa"],
        "arma": state.solucao["arma"],
        "local": state.solucao["local"],
        "pessoa_user": tentativa["pessoa"],
        "arma_user": tentativa["arma"],
        "local_user": tentativa["local"],
        "resultado_pessoa": "CORRETO" if "pessoa" in acertos else "ERRADO",
        "resultado_arma": "CORRETO" if "arma" in acertos else "ERRADO",
        "resultado_local": "CORRETO" if "local" in acertos else "ERRADO",
        "tentativa_num": state.tentativas,
        "max_tentativas": state.max_tentativas,
        "restantes": restantes,
        "historico": state.resumo_historico(),
    })

    return resposta.content


# ==============================
# REGRA DE DECISAO
# ==============================
# CORRECAO: comparacao normalizada (lowercase + strip) para nao falhar
# em "Mordomo" vs "mordomo" ou " faca " vs "faca".
# Tambem retorna acertos separadamente para o Agente 3 usar.

def verificar_tentativa(solucao, tentativa):
    erros = []
    acertos = []

    for campo in ["pessoa", "arma", "local"]:
        if tentativa[campo].strip().lower() == solucao[campo].strip().lower():
            acertos.append(campo)
        else:
            erros.append(campo)

    resultado = "acertou" if not erros else "errou"
    return resultado, erros, acertos


# ==============================
# MOTOR DO JOGO
# ==============================

PERSONAGENS = ["mordomo", "herdeira", "jardineiro"]
ARMAS = ["veneno", "faca", "corda"]
LOCAIS = ["biblioteca", "cozinha", "jardim"]


def jogar():
    print("\n" + "="*50)
    print("  Murdle - MISTERIO NA MANSAO")
    print("="*50)
    print("\nGerando misterio...\n")

    state = EstadoJogo()
    criar_historia(state)

    print("HISTORIA")
    print("-"*40)
    print(state.historia)
    print("-"*40)
    print(f"\nSuspeitos : {', '.join(PERSONAGENS)}")
    print(f"Armas     : {', '.join(ARMAS)}")
    print(f"Locais    : {', '.join(LOCAIS)}")
    print(f"\nVoce tem {state.max_tentativas} tentativas de acusacao.")

    while True:
        print("\n" + "="*40)
        print(f"Tentativas: {state.tentativas}/{state.max_tentativas} | "
              f"Interrogatorios: {len(state.historico_interrogatorios)}")
        print("-"*40)
        print("1 - Interrogar suspeito")
        print("2 - Fazer acusacao")
        print("3 - Ver historico de interrogatorios")
        print("4 - Sair")

        escolha = input("\nEscolha: ").strip()

        # Interrogar
        if escolha == "1":
            print(f"\nSuspeitos: {', '.join(PERSONAGENS)}")
            personagem = input("Quem interrogar? ").strip().lower()

            if personagem not in PERSONAGENS:
                print(f"Invalido. Escolha entre: {', '.join(PERSONAGENS)}")
                continue

            pergunta = input("Sua pergunta: ").strip()
            if not pergunta:
                print("Digite uma pergunta.")
                continue

            print("\nAguardando resposta...")
            resposta = perguntar(personagem, pergunta, state)
            print(f'\n{personagem.upper()} diz:\n"{resposta}"')

        # Acusar
        elif escolha == "2":
            if state.tentativas >= state.max_tentativas:
                print("\nSem tentativas restantes!")
                break

            print(f"\nSuspeitos: {', '.join(PERSONAGENS)}")
            pessoa = input("Quem e o culpado? ").strip()
            print(f"Armas: {', '.join(ARMAS)}")
            arma = input("Qual arma? ").strip()
            print(f"Locais: {', '.join(LOCAIS)}")
            local = input("Qual local? ").strip()

            tentativa = {"pessoa": pessoa, "arma": arma, "local": local}
            resultado, erros, acertos = verificar_tentativa(state.solucao, tentativa)

            if resultado == "acertou":
                print("\nVOCE RESOLVEU O MISTERIO!")
                print(f"\nSolucao: {state.solucao}")
                print(f"Tentativas usadas  : {state.tentativas + 1}")
                print(f"Interrogatorios    : {len(state.historico_interrogatorios)}")
                break

            state.tentativas += 1
            print(f"\nErrado.")
            print(f"   Acertos : {acertos if acertos else 'nenhum'}")
            print(f"   Erros   : {erros}")

            if state.tentativas >= state.max_tentativas:
                print("\nFIM DE JOGO. O culpado escapou.")
                print(f"Solucao correta: {state.solucao}")
                break

            print("\nGerando dica...")
            dica = gerar_dica(state, tentativa, erros, acertos)
            print(f"\nInvestigador diz:\n{dica}")

        # Historico
        elif escolha == "3":
            print("\nHISTORICO DE INTERROGATORIOS")
            print("-"*40)
            print(state.resumo_historico())

        elif escolha == "4":
            print("\nAte a proxima, detetive.")
            break

        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    jogar()