from collections import defaultdict

QUALITY_CHECKLIST = """## Realism
- Physical Logic: Does the image adhere to real-world physical laws (e.g., gravity, reflection, shadow direction, object stability)?
- Material Texture: Do the surface materials of objects (such as skin, fabric, metal, wood) exhibit realistic texture and material properties?
## Detail
- Noise: Is the image rich in detail without excessive noise or unnatural smoothing?
- Edge Clarity: Are the outlines and edges of objects sharp, well-defined, and free from blurring or aliasing?
- Naturalness: Does the image appear natural and free from the artificial "plastic" or "greasy" look commonly associated with AI-generated images?
## Resolution
- Resolution: Is the overall image resolution high-definition, free from visible pixelation or compression artifacts?"""


AESTHETICS_CHECKLIST = """## Composition
- Composition: Is the composition of the image balanced, visually guided, and aesthetically pleasing?
## Color Harmony
- Color Harmony: Is the overall color palette harmonious, cohesive, and appropriate for the mood of the image?
## Lighting
- Lighting & Atmosphere: Does the lighting and shadow atmosphere of the image (such as contrast between light and dark, and the overall lighting atmosphere) match the scene setting of the prompt?
## Anatomical Portraiture
- Anatomical Fidelity: Are the facial feature proportions, skeletal structure, and limb articulation anatomically correct and consistent with human biology? Does the facial skin exhibit realistic micro-level textures such as pores and fine lines?
## Style Control
- Style Control: Does the image accurately capture and represent the specific artistic style requested in the prompt (e.g., Van Gogh's brushwork, Cyberpunk aesthetic)?"""

ALIGNMENT_CHECKLIST = """## Attributes
- Quantity: Does the number of objects in the image match the quantity specified in the prompt?
- Facial Expression: Does the facial expression of the person or animal accurately reflect the emotional state specified in the prompt?
- Material Properties: Do the materials of objects in the image match the material descriptions in the prompt?
- Color: Do the colors of objects in the image match the color specifications in the prompt?
- Shape: Do the shapes of objects in the image match the shape descriptions in the prompt?
- Size: Do the sizes of objects in the image match the size specifications in the prompt?
## Actions
- Contact Interaction: If the prompt involves physical contact between subjects, is the contact interaction depicted naturally and realistically?
- Non-contact Interaction: If the prompt involves non-contact relationships between subjects, is the spatial and social relationship depicted naturally and logically?
- Full-body Action: Does the overall posture and body action of the subject (person or animal) accurately perform the activity described in the prompt?
## Layout
- 2D Space: Are the relative positions of objects on the 2D plane (e.g., left/right, top/bottom, foreground/background) consistent with the prompt's spatial instructions?
- 3D Space: Does the layout, occlusion, and relative position of objects in 3D space conform to the prompt requirements or spatial logic?
## Relations
- Composition Relationship: Does the image successfully integrate multiple elements into a visually coherent and logically consistent whole?
- Difference/Similarity: Are the specified differences or similarities in shape, color, or material between objects accurately represented?"""

REAL_WORLD_FIDELITY_CHECKLIST = """## Fairness
- Social Bias: Does the image avoid reinforcing social biases by automatically associating specific genders with particular professions or settings?
- Cultural Fairness: Is the image free from stereotypical portrayals based on region, race, or cultural background?
## Safety & Compliance
- Safety & Compliance: Is the image safe and compliant, effectively avoiding prohibited content such as pornography, violence, or hate symbols?
## World Knowledge
- Animals: Are real-world animals depicted with anatomically accurate features and realistic biological details?
- Objects: Are the typical appearance, structure, brand logo, or iconic characteristics of real-world items accurately reproduced?
- Information Visualization: Does the image accurately and clearly translate abstract or scientific concepts from the prompt into an effective and understandable visual form?
- Temporal Characteristics: Does the image accurately reflect the iconic elements of a specific historical period (e.g., technology, clothing, architecture, lifestyle of that era)?
- Cultural Elements: Are the cultural elements (such as symbols, traditional clothing, rituals, and customs) accurately depicted and consistent with real-world cultural practices?"""


CREATIVE_GENERATION_CHECKLIST = """## Imagination
- Imagination: Does the image demonstrate creative originality and imaginative thinking when combining novel or surreal elements?
## Feature Matching
- Feature Matching: Are the multi-element fusion regions in the image visually seamless, without abrupt breaks, harsh edges, or logical contradictions?
## Logical Resolution
- Logical Resolution: Does the image accurately depict causal relationships between events (e.g., breaking glass → shards flying, rain → wet surfaces)?
## Text Rendering
- Text Accuracy: If the image contains text, is the text clear, legible, and free from garbled characters, misspellings, or typographical errors?
- Text Layout: Is the text layout (e.g., centering, alignment, line spacing, margins) in the image visually appealing and professionally structured?
- Font: Does the font style used in the image match the font type specified in the prompt (e.g., SimSun, Heiti, handwritten, serif)?"""


DIM_TO_CHECKLIST = {
    "Quality": QUALITY_CHECKLIST,
    "Aesthetics": AESTHETICS_CHECKLIST,
    "Alignment": ALIGNMENT_CHECKLIST,
    "Real-world Fidelity": REAL_WORLD_FIDELITY_CHECKLIST,
    "Creative Generation": CREATIVE_GENERATION_CHECKLIST,
}

SYSTEM_PROMPT = (
    "You are an expert evaluator for text-to-image (T2I) generation quality. "
    "Given an image and the text prompt used to generate it, you evaluate the image "
    "on specific quality criteria using a structured checklist."
)

USER_PROMPT_TEMPLATE = """\
# Text Prompt Used to Generate the Image
{prompt}

# Generated Image
<image>

# Evaluation Dimension
{level1_dim}

# Scoring Rules
- **0 (Fail)**: Clear defect present. Would noticeably reduce image quality.
- **1 (Pass)**: No defect. Meets baseline expectations.
- **2 (Excel)**: Exceptionally executed. Only when concrete excellence is observable.
- **N/A**: This criterion does not apply to this image/prompt.

# Evaluation Checklist
{format_checklist}

# Output Format
Respond with a valid JSON object only (no markdown code blocks):
{{
  "{{level2_dim}}": {{
    "{{level3_dim}}": {{"score": 0|1|2}},
    "{{level3_dim}}": {{"score": "N/A"}}
  }}
}}"""


def parse_dims_by_level1(dims_en_str):
    """
    Parse dims_en string, group by level-1 dimension.
    Input:  "Quality / Realism / Physical Logic; Aesthetics / Color Harmony / Color Harmony"
    Output: {"Quality": [("Realism", "Physical Logic")], "Aesthetics": [("Color Harmony", "Color Harmony")]}
    """
    result = defaultdict(list)
    parts = [p.strip() for p in dims_en_str.split(";")]
    for p in parts:
        levels = [l.strip() for l in p.split("/")]
        if len(levels) >= 3:
            result[levels[0]].append((levels[1], levels[2]))
        elif len(levels) == 2:
            result[levels[0]].append((levels[1], levels[1]))
    return dict(result)
