"""Clients LLM avec appel d'outils : Gemini (API, cle GEMINI_API_KEY) ou Ollama (local).

Fournisseur : LLM_PROVIDER=gemini|ollama ; par defaut Gemini si une cle est definie, sinon Ollama.
Meme boucle pour les deux : le modele demande un outil, on l'execute, on lui renvoie le resultat,
jusqu'a une reponse texte (3 tours au plus).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

Executeur = Callable[[str, dict], dict]
MAX_TOURS = 3


@dataclass
class Sortie:
    texte: str
    appels: list[dict] = field(default_factory=list)  # [{"outil", "arguments", "resultat"}]


PENSEE_RE = re.compile(r"<think>.*?</think>", re.S)


def sans_pensee(texte: str) -> str:
    """Retire un bloc <think>...</think> que certains modeles laissent dans la reponse."""
    return PENSEE_RE.sub("", texte or "").strip()


class Ollama:
    """Modele local. Les modeles a "reflexion" (Qwen3.5...) l'ont desactivee par defaut :
    reponses plus courtes et plus rapides ; OLLAMA_THINK=1 pour la garder."""

    def __init__(self, modele: str | None = None, url: str | None = None):
        self.modele = modele or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.nom = f"ollama:{self.modele}"
        self._reflexion: bool | None = None

    def _post(self, chemin: str, corps: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{chemin}", data=json.dumps(corps).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def sait_reflechir(self) -> bool:
        if self._reflexion is None:
            infos = self._post("/api/show", {"model": self.modele})
            self._reflexion = "thinking" in infos.get("capabilities", [])
        return self._reflexion

    def _chat(self, messages: list[dict], outils: list[dict]) -> dict:
        corps = {
            "model": self.modele, "messages": messages, "stream": False,
            "tools": [{"type": "function", "function": o} for o in outils],
            # num_predict : une reponse fait 150 mots au plus ; la limite coupe un modele qui s'emballe.
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
        }
        if self.sait_reflechir():
            corps["think"] = os.getenv("OLLAMA_THINK", "") == "1"
        return self._post("/api/chat", corps)["message"]

    def repondre(self, systeme: str, message: str, outils: list[dict], executer: Executeur) -> Sortie:
        messages = [{"role": "system", "content": systeme}, {"role": "user", "content": message}]
        appels: list[dict] = []
        for _ in range(MAX_TOURS):
            m = self._chat(messages, outils)
            demandes = m.get("tool_calls") or []
            if not demandes:
                return Sortie(sans_pensee(m.get("content", "")), appels)
            messages.append(m)
            for d in demandes:
                nom, args = d["function"]["name"], d["function"].get("arguments") or {}
                resultat = executer(nom, args)
                appels.append({"outil": nom, "arguments": args, "resultat": resultat})
                messages.append({"role": "tool", "tool_name": nom, "content": json.dumps(resultat, ensure_ascii=False)})
        return Sortie(sans_pensee(self._chat(messages, [])["content"]), appels)

    def extraire(self, systeme: str, texte: str, schema: dict) -> dict:
        """Reponse JSON forcee par le schema (decodage contraint d'Ollama). {} si illisible.
        Meme num_ctx que _chat : en changer obligerait Ollama a recharger le modele."""
        corps = {
            "model": self.modele, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": systeme}, {"role": "user", "content": texte}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 256},
        }
        if self.sait_reflechir():
            corps["think"] = False
        contenu = sans_pensee(self._post("/api/chat", corps)["message"].get("content", ""))
        try:
            valeur = json.loads(contenu)
        except json.JSONDecodeError:
            return {}
        return valeur if isinstance(valeur, dict) else {}


class Gemini:
    def __init__(self, modele: str | None = None):
        from google import genai

        self.modele = modele or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.nom = f"gemini:{self.modele}"

    def repondre(self, systeme: str, message: str, outils: list[dict], executer: Executeur) -> Sortie:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=systeme,
            temperature=0,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=o["name"], description=o["description"],
                                          parameters_json_schema=o["parameters"])
                for o in outils
            ])] if outils else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contenus = [types.Content(role="user", parts=[types.Part.from_text(text=message)])]
        appels: list[dict] = []
        for _ in range(MAX_TOURS):
            r = self.client.models.generate_content(model=self.modele, contents=contenus, config=config)
            demandes = r.function_calls or []
            if not demandes:
                return Sortie((r.text or "").strip(), appels)
            contenus.append(r.candidates[0].content)
            reponses = []
            for d in demandes:
                args = dict(d.args or {})
                resultat = executer(d.name, args)
                appels.append({"outil": d.name, "arguments": args, "resultat": resultat})
                reponses.append(types.Part.from_function_response(name=d.name, response=resultat))
            contenus.append(types.Content(role="user", parts=reponses))
        config.tools = None
        r = self.client.models.generate_content(model=self.modele, contents=contenus, config=config)
        return Sortie((r.text or "").strip(), appels)

    def extraire(self, systeme: str, texte: str, schema: dict) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=systeme, temperature=0,
            response_mime_type="application/json", response_json_schema=schema,
        )
        r = self.client.models.generate_content(model=self.modele, contents=texte, config=config)
        try:
            valeur = json.loads(r.text or "{}")
        except json.JSONDecodeError:
            return {}
        return valeur if isinstance(valeur, dict) else {}


def choisir_llm():
    fournisseur = os.getenv("LLM_PROVIDER") or ("gemini" if os.getenv("GEMINI_API_KEY") else "ollama")
    return Gemini() if fournisseur == "gemini" else Ollama()
