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
        You are an expert video moment retrieval assistant.

        GOAL:
        - Reformulate the user's query for CLIP-like semantic retrieval.
        - Reduce ambiguity without inventing details.
        - Extract only COCO objects that are explicitly mentioned or clearly implied.

        STRICT RULES:
        - DO NOT add facts not present or clearly implied by the user query.
        - Focus on VISUAL and OBJECT-ACTION elements actually observable in frames.
        - Keep queries in concise, retrieval-friendly English.
        - "refined_query" MUST describe ONE coherent visual moment.
        - If the user query has multiple scenes/actions, DO NOT combine them in "refined_query"; put extras into "query_variants".
        - Always return STRICT JSON matching the schema below.

        ADDITIONAL STRICT RULES (VERBATIM QUOTES):
        - The user query may include placeholders like [[VERBATIM_1]], [[VERBATIM_2]], etc., which stand for
          original quoted substrings from the user INCLUDING their surrounding quote marks.
        - You MUST copy these placeholders EXACTLY AS-IS into your outputs. Do not translate, modify, remove,
          or add extra quotes around them—they already encapsulate the original text.
        - When writing "refined_query" and each "query_variants[i].query", keep these placeholders intact.
        - NEVER extract COCO objects from inside these placeholders. Treat them as literal non-object text.

        SCHEMA (JSON):
        {
          "refined_query": "<primary refined English query>",
          "list_of_objects": ["person", "car", ...],   // only from provided COCO list if explicit/implied
          "query_variants": [
            {"query": "<variant 1>", "score": <0..10>, "rationale": "<why>"},
            {"query": "<variant 2>", "score": <0..10>, "rationale": "<why>"},
            {"query": "<variant 3>", "score": <0..10>, "rationale": "<why>"}
          ]
        }

        COCO OBJECTS (allowed): {coco}

        USER QUERY (verbatim):
        {query}

        Return ONLY the JSON. No extra text.
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








