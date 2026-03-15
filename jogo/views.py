"""
views.py — Views Django do jogo Murdle.
"""

import json
import os
from dotenv import load_dotenv

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET

from .agents import (
    EstadoJogo, criar_historia,
    interrogar_suspeito, gerar_dica, verificar_tentativa,
)

# carrega variáveis do .env
load_dotenv()

SESSION_KEY = "murdle_estado"


def _get_state(request):
    data = request.session.get(SESSION_KEY)
    return EstadoJogo.from_dict(data) if data else None


def _save_state(request, state: EstadoJogo):
    request.session[SESSION_KEY] = state.to_dict()
    request.session.modified = True


# ── Tela inicial ─────────────────────────────────────────

def index(request):
    return render(request, "game/index.html")


# ── Criar novo jogo ──────────────────────────────────────

@require_POST
def novo_jogo(request):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return render(
            request,
            "game/index.html",
            {"erro": "GOOGLE_API_KEY não encontrada no .env"}
        )

    # garante que a key esteja disponível para os agentes
    os.environ["GOOGLE_API_KEY"] = api_key

    state = EstadoJogo()

    try:
        criar_historia(state)
    except Exception as e:
        return render(request, "game/index.html", {"erro": str(e)})

    _save_state(request, state)
    return redirect("jogo")


# ── Tela principal ───────────────────────────────────────

@require_GET
def jogo(request):
    state = _get_state(request)

    if not state:
        return redirect("index")

    ctx = {
        "historia_paragrafos": [p for p in state.historia.split("\n") if p.strip()],
        "historico": state.historico,
        "tentativas": state.tentativas,
        "max_tentativas": state.max_tentativas,
        "game_over": state.game_over,
        "vitoria": state.vitoria,
        "solucao": state.solucao if state.game_over else None,
        "personagens_lista": state.personagens,      # [{"id","nome","papel","genero"}]
        "personagens_nomes": [p["nome"] for p in state.personagens],  # para os selects
        "armas": EstadoJogo.ARMAS,
        "locais": EstadoJogo.LOCAIS,
        "dots": ["used" if i < state.tentativas else "remaining"
                for i in range(state.max_tentativas)],
    }

    return render(request, "game/jogo.html", ctx)


# ── AJAX: Interrogar ─────────────────────────────────────

@require_POST
def interrogar(request):

    state = _get_state(request)

    if not state or state.game_over:
        return JsonResponse(
            {"erro": "Jogo não encontrado ou encerrado."},
            status=400
        )

    try:
        body = json.loads(request.body)
        pid      = body.get("pid", "").strip()
        pergunta = body.get("pergunta", "").strip()
    except Exception:
        return JsonResponse({"erro": "Dados inválidos."}, status=400)

    if not pid:
        return JsonResponse({"erro": "Nenhum personagem selecionado."}, status=400)

    if not state.get_personagem(pid):
        return JsonResponse({"erro": f"Personagem inválido: {pid}"}, status=400)

    if not pergunta:
        return JsonResponse({"erro": "A pergunta não pode estar vazia."}, status=400)

    try:
        resposta   = interrogar_suspeito(pid, pergunta, state)
        personagem = state.get_personagem(pid)
    except Exception as e:
        return JsonResponse({"erro": str(e)}, status=500)

    _save_state(request, state)

    return JsonResponse({
        "pid":      pid,
        "nome":     personagem["nome"],
        "genero":   personagem["genero"],
        "pergunta": pergunta,
        "resposta": resposta,
    })


# ── AJAX: Acusar ─────────────────────────────────────────

@require_POST
def acusar(request):

    state = _get_state(request)

    if not state or state.game_over:
        return JsonResponse(
            {"erro": "Jogo não encontrado ou encerrado."},
            status=400
        )

    try:
        body = json.loads(request.body)

        pessoa = body.get("pessoa", "").strip()
        arma = body.get("arma", "").strip()
        local = body.get("local", "").strip()

    except Exception:
        return JsonResponse({"erro": "Dados inválidos."}, status=400)

    if not all([pessoa, arma, local]):
        return JsonResponse(
            {"erro": "Preencha todos os campos."},
            status=400
        )

    tentativa = {"pessoa": pessoa, "arma": arma, "local": local}

    resultado, erros, acertos = verificar_tentativa(
        state.solucao,
        tentativa
    )

    resp = {
        "resultado": resultado,
        "acertos": acertos,
        "erros": erros,
        "tentativas": state.tentativas,
        "max_tentativas": state.max_tentativas,
        "dica": None,
        "solucao": None,
    }

    if resultado == "acertou":

        state.tentativas += 1
        state.game_over = True
        state.vitoria = True

        resp["tentativas"] = state.tentativas
        resp["solucao"] = state.solucao

    else:

        state.tentativas += 1
        resp["tentativas"] = state.tentativas

        if state.tentativas >= state.max_tentativas:

            state.game_over = True
            state.vitoria = False
            resp["solucao"] = state.solucao

        else:

            try:
                resp["dica"] = gerar_dica(
                    tentativa,
                    erros,
                    acertos,
                    state
                )

            except Exception as e:
                resp["dica"] = f"(Erro ao gerar dica: {e})"

    _save_state(request, state)

    return JsonResponse(resp)


# ── Reiniciar ────────────────────────────────────────────

@require_POST
def reiniciar(request):

    request.session.pop(SESSION_KEY, None)

    return redirect("index")