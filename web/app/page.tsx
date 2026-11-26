"use client";

import { FormEvent, ReactNode, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Passage = {
  numero: number;
  reference: string;
  article_id: string;
  contexte: string;
  langue: string;
  texte: string;
  score: number;
  cite: boolean;
};

type Calcul = {
  outil: string;
  arguments: Record<string, unknown>;
  resultat: { resultat?: { montant: number; formule: string }; erreur?: string };
};

type Reponse = {
  reponse: string;
  refus: boolean;
  motif_refus: string | null;
  passages: Passage[];
  calculs: Calcul[];
  avertissement: string;
};

const DOMAINES = [
  { valeur: "", label: "Tous les domaines" },
  { valeur: "location", label: "Location" },
  { valeur: "vente", label: "Vente" },
  { valeur: "fiscalite", label: "Fiscalité" },
  { valeur: "copropriete", label: "Copropriété" },
];

const EXEMPLES = [
  "Mon propriétaire me demande trois mois de caution, c'est légal ?",
  "Je loue mon appartement 4 000 DH par mois. Combien d'impôt je paie ?",
  "Le syndic peut-il agir contre un voisin qui ne paie pas les charges ?",
  "Je voudrais sous-louer une chambre de mon appartement à un étudiant. J'ai le droit ?",
];

const OUTILS: Record<string, string> = {
  impot_profit_immobilier: "Impôt sur le profit immobilier",
  impot_revenus_fonciers: "Impôt sur les loyers",
  droits_enregistrement: "Droits d'enregistrement",
};

const dh = (n: number) =>
  `${n.toLocaleString("fr-FR", { maximumFractionDigits: 2 }).replace(/ /g, " ")} DH`;

function AvecCitations({ texte }: { texte: string }) {
  const morceaux: ReactNode[] = texte.split(/(\[S\d+\])/g).map((m, i) => {
    const n = m.match(/^\[S(\d+)\]$/);
    return n ? (
      <a key={i} className="cite" href={`#source-${n[1]}`} aria-label={`Source ${n[1]}`}>
        {n[1]}
      </a>
    ) : (
      m
    );
  });
  return <>{morceaux}</>;
}

function CartePassage({ p }: { p: Passage }) {
  const [ouvert, setOuvert] = useState(false);
  const long = p.texte.length > 420;
  return (
    <article id={`source-${p.numero}`} className={`passage ${p.cite ? "cite-oui" : ""}`}>
      <header>
        <span className="numero">S{p.numero}</span>
        <strong>{p.reference}</strong>
        {p.langue === "ar" && <span className="etiquette">texte arabe</span>}
      </header>
      {p.contexte && <p className="chemin">{p.contexte}</p>}
      <p className="texte" dir="auto">
        {ouvert || !long ? p.texte : `${p.texte.slice(0, 420)}...`}
      </p>
      {long && (
        <button type="button" className="lien" onClick={() => setOuvert(!ouvert)}>
          {ouvert ? "Réduire" : "Lire l'article en entier"}
        </button>
      )}
    </article>
  );
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [domaine, setDomaine] = useState("");
  const [chargement, setChargement] = useState(false);
  const [erreur, setErreur] = useState<string | null>(null);
  const [reponse, setReponse] = useState<Reponse | null>(null);

  async function envoyer(e?: FormEvent, texte?: string) {
    e?.preventDefault();
    const q = (texte ?? question).trim();
    if (q.length < 3) return;
    setQuestion(q);
    setChargement(true);
    setErreur(null);
    setReponse(null);
    try {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, domaine: domaine || null }),
      });
      if (!r.ok) throw new Error(`erreur ${r.status}`);
      setReponse(await r.json());
    } catch {
      setErreur(`Le serveur ne répond pas (${API}). Vérifiez que l'API est lancée.`);
    } finally {
      setChargement(false);
    }
  }

  const cites = reponse?.passages.filter((p) => p.cite) ?? [];
  const autres = reponse?.passages.filter((p) => !p.cite) ?? [];

  return (
    <main>
      <header className="entete">
        <h1>Dar en Clair</h1>
        <p>
          Location, vente, fiscalité immobilière et copropriété au Maroc. Les réponses viennent des
          textes de loi et citent leurs articles.
        </p>
      </header>

      <form onSubmit={envoyer} className="formulaire">
        <label htmlFor="question" className="cache">
          Votre question
        </label>
        <textarea
          id="question"
          dir="auto"
          rows={3}
          maxLength={1000}
          placeholder="Ex. : mon propriétaire peut-il augmenter le loyer chaque année ?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) envoyer(e);
          }}
        />
        <div className="ligne">
          <select value={domaine} onChange={(e) => setDomaine(e.target.value)} aria-label="Domaine">
            {DOMAINES.map((d) => (
              <option key={d.valeur} value={d.valeur}>
                {d.label}
              </option>
            ))}
          </select>
          <button type="submit" disabled={chargement || question.trim().length < 3}>
            {chargement ? "Recherche..." : "Demander"}
          </button>
        </div>
      </form>

      {!reponse && !chargement && !erreur && (
        <section className="exemples">
          <h2>Exemples</h2>
          {EXEMPLES.map((x) => (
            <button key={x} type="button" dir="auto" onClick={() => envoyer(undefined, x)}>
              {x}
            </button>
          ))}
        </section>
      )}

      {chargement && <p className="attente">Recherche dans les textes, puis rédaction de la réponse...</p>}
      {erreur && <p className="erreur">{erreur}</p>}

      {reponse && (
        <section className="resultat">
          <div className={`reponse ${reponse.refus ? "refus" : ""}`} dir="auto">
            <AvecCitations texte={reponse.reponse} />
          </div>

          {reponse.calculs.length > 0 && (
            <div className="calculs">
              {reponse.calculs.map((c, i) => (
                <div key={i} className="calcul">
                  <span>{OUTILS[c.outil] ?? c.outil}</span>
                  {c.resultat.resultat ? (
                    <strong>{dh(c.resultat.resultat.montant)}</strong>
                  ) : (
                    <em>{c.resultat.erreur}</em>
                  )}
                </div>
              ))}
              <p className="note">Calcul fait par du code testé à partir des taux du CGI 2026, pas par le modèle.</p>
            </div>
          )}

          <p className="avertissement">{reponse.avertissement}</p>

          {cites.length > 0 && (
            <>
              <h2>Articles cités</h2>
              {cites.map((p) => (
                <CartePassage key={p.numero} p={p} />
              ))}
            </>
          )}
          {autres.length > 0 && (
            <details>
              <summary>Autres passages consultés ({autres.length})</summary>
              {autres.map((p) => (
                <CartePassage key={p.numero} p={p} />
              ))}
            </details>
          )}
        </section>
      )}

      <footer>
        Lois 67-12, 49-16, 39-08, 32-09, 18-00, 47-06 et Code général des impôts 2026.{" "}
        <a href="https://github.com/CrimsonAzul/dar-en-clair">Code source</a>
      </footer>
    </main>
  );
}
