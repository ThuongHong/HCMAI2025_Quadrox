from llama_index.core.prompts import PromptTemplate

COCO_CLASS = """
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
"""

class Prompt:

    VISUAL_EVENT_EXTRACTION_PROMPT = PromptTemplate(
        """
        You are an expert video moment retrieval system. Extract and optimize the query for semantic video search.
        
        COCO Objects Available: {coco}
        
        Original Query: {query}
        
        Your task:
        1. Extract key visual elements, actions, and temporal cues from the ORIGINAL query
        2. Create a refined query that PRESERVES the user's intent while optimizing for search
        3. Generate semantic variations that EXPAND coverage without over-specifying
        4. Identify relevant COCO objects ONLY if they are explicitly mentioned or clearly implied
        
        CRITICAL GUIDELINES:
        - PRESERVE the user's original intent - do not add details not present in the query
        - Focus on VISUAL elements that are observable in video frames
        - Use the refined query as a SEARCH OPTIMIZATION, not a complete rewrite
        - Generate variations that are SEMANTICALLY RELATED but not overly specific
        - Only suggest objects that are EXPLICITLY mentioned or clearly implied
        - Avoid "hallucinating" details like specific settings, lighting, or environmental context unless stated
        
        Example Good Approach:
        Query: "a woman places a picture and drives to store"
        Refined: "woman placing picture frame, person driving car"
        Variations: 
        - "person handling picture frame object"
        - "woman driving vehicle transportation"
        - "picture frame placement activity"
        
        Example Bad Approach (AVOID):
        Query: "a woman places a picture and drives to store"
        Refined: "woman hanging framed picture on wall indoor home setting, woman driving car vehicle outdoor road"
        (This adds too many specific details not in the original query)
        
        Return:
        - refined_query: Optimized search query that preserves user intent
        - list_of_objects: Relevant COCO objects (only if explicitly mentioned or clearly implied)
        - query_variations: 3-4 semantic variations for comprehensive search coverage
        """
    )


    ANSWER_GENERATION_PROMPT = PromptTemplate(
    """
    Based on the user's query and the relevant keyframes found, generate a comprehensive answer.
    
    Original Query and questions: {original_query}
    
    Relevant Keyframes:
    {keyframes_context}
    
    Please provide a detailed answer that:
    1. Directly addresses the user's query
    2. References specific information from the keyframes
    3. Synthesizes information across multiple keyframes if relevant
    4. Mentions which videos/keyframes contain the most relevant content
    
    Keep the answer informative but concise.
    """
    )








