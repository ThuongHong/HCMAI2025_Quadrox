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



class VisualEventExtractor:
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.extraction_prompt = Prompt.VISUAL_EVENT_EXTRACTION_PROMPT
            

    async def extract_visual_events(self, query: str) -> AgentResponse:
        prompt = self.extraction_prompt.format(query=query, coco=COCO_CLASS)
        response = await self.llm.as_structured_llm(AgentResponse).acomplete(prompt)
        obj = cast(AgentResponse, response.raw)
        return obj
    

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







