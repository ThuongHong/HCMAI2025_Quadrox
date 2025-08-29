# service/tc_grounding_en.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re
import numpy as np

@dataclass
class TCGroundingResult:
    refined_query: str
    targets: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    events: List[Dict[str, List[str]]] = field(default_factory=list)  # [{targets:[], contexts:[], actions:[]}, ...]

class CanonicalVocab:
    def __init__(self, model_service, objects: List[str], places: List[str],
                 contexts: List[str], actions: List[str], normalize=True):
        self.ms = model_service
        self.objects = objects
        self.places  = places
        self.contexts= contexts
        self.actions = actions
        self.normalize = normalize
        self.lex = {
            "object": self.objects,
            "place" : self.places,
            "context": self.contexts,
            "action": self.actions
        }
        # Precompute embeddings for all canonical terms (CPU)
        self.emb = {k: self.ms.embedding_many_texts(v) for k, v in self.lex.items()}

    def nearest(self, phrase: str, kind: str, thr: float=0.40) -> Optional[str]:
        e = self.ms.embedding_text(phrase)  # (d,) normalized
        M = self.emb[kind]                 # (n, d)
        sims = M @ e                        # cosine
        j = int(np.argmax(sims))
        if float(sims[j]) >= thr:
            return self.lex[kind][j]
        return None

class TCGrounderEnglish:
    def __init__(self, model_service, canon: CanonicalVocab):
        self.ms = model_service
        self.canon = canon
        # light keyword sets for contexts
        self.colors = {"red","green","blue","yellow","black","white","purple","orange","brown","pink"}
        self.time_ctx = {"daytime","night","dawn","dusk","evening","noon"}
        self.view_ctx = {"aerial","drone","overhead","handheld","zoom_in","zoom_out"}
        self.scene_ctx= {"festival","stage","crowd","interview","banner","ornament","backdrop","market","street","kitchen"}

    def _split_events(self, text: str) -> List[str]:
        # split by markers: E1:, then, after that, next, subsequently
        parts = re.split(r"(?:E\d+:|then|after that|next|subsequently|following that|and then)", text, flags=re.I)
        # rejoin tiny fragments
        chunks = [p.strip(" ,.;") for p in parts if p and p.strip(" ,.;")]
        return chunks if len(chunks) >= 1 else [text]

    def extract(self, query_en: str) -> TCGroundingResult:
        # 1) Split into events
        chunks = self._split_events(query_en)
        all_T, all_C, all_A = [], [], []
        events = []

        for ch in chunks:
            # naive NP/action extraction by regex+lists (no spaCy dependency)
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", ch)
            cand_nouns = []  # will be canonicalized to object/place
            cand_actions= []
            cand_context= []

            for t in tokens:
                lt = t.lower()
                if lt in self.colors or lt in self.time_ctx or lt in self.view_ctx or lt in self.scene_ctx:
                    cand_context.append(lt)
                # crude verb heuristic
                if lt in {"cut","flip","lift","sprinkle","answer","interview","hold","raise","slice"}:
                    cand_actions.append(lt)
                # nouns candidates: keep all non-stopwords simple
                if lt not in {"the","a","an","and","or","then","after","that","first","second","third"}:
                    cand_nouns.append(lt)

            # canonicalize nouns into object/place
            T_local, C_local, A_local = [], [], []
            for n in cand_nouns:
                hit = self.canon.nearest(n, "object") or self.canon.nearest(n, "place")
                if hit:
                    T_local.append(hit)
            # canonicalize contexts/actions
            for c in set(cand_context):
                hitc = self.canon.nearest(c, "context") or c
                C_local.append(hitc)
            for a in set(cand_actions):
                hita = self.canon.nearest(a, "action") or a
                A_local.append(hita)

            # dedup preserve order
            def uniq(xs): 
                s, o = set(), []
                for x in xs:
                    if x not in s:
                        s.add(x); o.append(x)
                return o
            T_local, C_local, A_local = uniq(T_local), uniq(C_local), uniq(A_local)

            events.append({"targets": T_local, "contexts": C_local, "actions": A_local})
            all_T.extend(T_local); all_C.extend(C_local); all_A.extend(A_local)

        # merge global lists with cap
        def capuniq(xs, k): 
            s, o = set(), []
            for x in xs:
                if x not in s:
                    s.add(x); o.append(x)
            return o[:k]
        return TCGroundingResult(
            refined_query=query_en.strip(),
            targets=capuniq(all_T, 6),
            contexts=capuniq(all_C, 4),
            actions=capuniq(all_A, 6),
            events=events
        )
