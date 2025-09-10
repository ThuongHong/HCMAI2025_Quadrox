import re
from typing import  cast
from llama_index.core.llms import LLM
from schema.agent import AgentResponse
from pathlib import Path

from typing import Dict, List, Tuple
from collections import defaultdict
from schema.response import KeyframeServiceReponse
import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from .promts import Prompt, COCO_CLASS



# --- Verbatim quoted text helpers (ASCII "..." and curly “ … ”) ---
from typing import Dict, Tuple

_QUOTE_PATTERNS = [
    (r'"([^"]+)"', '"', '"'),      # ASCII quotes
    (r'“([^”]+)”', '“', '”'),      # Curly quotes
]

def _preserve_verbatim_quoted(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace quoted substrings with [[VERBATIM_i]] placeholders.
    Return (processed_text, mapping). Mapping values INCLUDE original quotes.
    """
    mapping: Dict[str, str] = {}
    work = text
    counter = 1
    changed = True
    while changed:
        changed = False
        for pat, lq, rq in _QUOTE_PATTERNS:
            def _sub(m):
                nonlocal counter, changed
                inner = m.group(1)
                original = f"{lq}{inner}{rq}"  # keep quotes exactly
                token = f"[[VERBATIM_{counter}]]"
                counter += 1
                mapping[token] = original
                changed = True
                return token
            work = re.sub(pat, _sub, work)
    return work, mapping

def _restore_verbatim_tokens(text: str, mapping: Dict[str, str]) -> str:
    if not isinstance(text, str) or not mapping:
        return text
    out = text
    for token, original in mapping.items():
        out = out.replace(token, original)
    return out



class VisualEventExtractor:
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.extraction_prompt = Prompt.VISUAL_EVENT_EXTRACTION_PROMPT
            

    async def extract_visual_events(self, query: str) -> AgentResponse:
        import json
        # PRE: protect quoted phrases
        protected_query, verb_map = _preserve_verbatim_quoted(query)

        prompt = self.extraction_prompt.format(query=protected_query, coco=COCO_CLASS)
        try:
            resp = await self.llm.achat(prompt)
            txt = resp.message.content if hasattr(resp, "message") else str(resp)

            # Strict parse -> substring fallback
            try:
                data = json.loads(txt)
            except Exception:
                start = txt.find("{"); end = txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(txt[start:end+1])
                else:
                    raise

            # POST: restore placeholders in outputs
            if isinstance(data.get("refined_query"), str):
                data["refined_query"] = _restore_verbatim_tokens(data["refined_query"], verb_map)

            if isinstance(data.get("query_variants"), list):
                fixed = []
                for v in data["query_variants"]:
                    if isinstance(v, dict):
                        q = v.get("query")
                        if isinstance(q, str):
                            v["query"] = _restore_verbatim_tokens(q, verb_map)
                        r = v.get("rationale")
                        if isinstance(r, str):
                            v["rationale"] = _restore_verbatim_tokens(r, verb_map)
                    fixed.append(v)
                data["query_variants"] = fixed

            # Safety: list_of_objects should not contain placeholders; drop any if present
            if isinstance(data.get("list_of_objects"), list):
                data["list_of_objects"] = [
                    o for o in data["list_of_objects"]
                    if not (isinstance(o, str) and o.startswith("[[VERBATIM_"))
                ]

            return AgentResponse(**data)

        except Exception:
            # Fallback: keep original query verbatim
            return AgentResponse(refined_query=query, list_of_objects=[], query_variants=[])
    

    @staticmethod
    def calculate_video_scores(keyframes: List[KeyframeServiceReponse]) -> List[Tuple[float, List[KeyframeServiceReponse]]]:
        """
        Calculate average scores for each video and return sorted by score
        
        Returns:
            List of tuples: (video_num, average_score, keyframes_in_video)
        """
        video_keyframes: Dict[str, List[KeyframeServiceReponse]] = defaultdict(list)
        
        for keyframe in keyframes:
            video_keyframes[f"{keyframe.group_num}/{keyframe.video_num}"].append(keyframe)
        
        video_scores: List[Tuple[float, List[KeyframeServiceReponse]]] = []
        for _, video_keyframes_list in video_keyframes.items():
            avg_score = sum(kf.confidence_score for kf in video_keyframes_list) / len(video_keyframes_list)
            video_scores.append((avg_score, video_keyframes_list))
        
        video_scores.sort(key=lambda x: x[0], reverse=True)
        
        return video_scores
    



class AnswerGenerator:
    """Generates final answers based on refined keyframes"""
    
    def __init__(self, llm: LLM, data_folder: str):
        self.data_folder = data_folder
        self.llm = llm
        self.answer_prompt = Prompt.ANSWER_GENERATION_PROMPT
    
    async def generate_answer(
        self,
        original_query: str,
        final_keyframes: List[KeyframeServiceReponse],
        objects_data: Dict[str, List[str]],
        
    ):
        chat_messages = []
        for kf in final_keyframes:
            keyy = f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg"
            objects = objects_data.get(keyy, [])

            image_path = os.path.join(self.data_folder, f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{kf.video_num:03d}/{kf.keyframe_num:03d}.jpg")

            context_text = f"""
            Keyframe {kf.key} from Video {kf.video_num} (Confidence: {kf.confidence_score:.3f}):
            - Detected Objects: {', '.join(objects) if objects else 'None detected'}
            """

            if os.path.exists(image_path):
                message_content = [
                    ImageBlock(path=Path(image_path)),
                    TextBlock(text=context_text)
                ]   
            else:
                message_content = [TextBlock(text=context_text + "\n(Image not available)")]
            
            user_message = ChatMessage(
                role=MessageRole.USER,
                content=message_content
            )

            chat_messages.append(user_message)

        
        final_prompt = self.answer_prompt.format(
            query=original_query,
            keyframes_context="See the keyframes and their context above"
        ) 
        query_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)

        response = await self.llm.achat(chat_messages)
        return response.message.content







