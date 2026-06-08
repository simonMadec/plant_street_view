import os

os.environ["HF_HOME"] = "/data/data2/hf_cache"

import json
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "/data/data2/plant_street_view/result/fusion_plantnet_split/figures/only_plot/9883094991815032_right.jpg"

# ### Savanna (natural, not cultivated)
# - 41 — Herbaceous Savanna (Savane Herbacee): dominated by tall grasses, very few or no trees/shrubs
# - 42 — Shrub Savanna (Savane Arbustive): grasses with scattered shrubs (<5 m), open landscape
# - 43 — Tree/Wooded Savanna (Savane Arboree): grasses with scattered mature trees (>5 m), open canopy
# ### Other Natural / Bare
# - 44 — Forest & Riparian forest (Foret et Ripisylve): dense closed tree canopy, often along rivers
# - 45 — Bare Soil (Sol Nu): exposed earth, no or negligible vegetation
# - 46 — Burned area (Brulis): blackened ground, charred vegetation, recent fire marks
# - 25 — Agroforestry (Agroforesterie): crops mixed with scattered useful trees (e.g. karité, néré)

# ### Fallow (previously cultivated, now resting)
# - 31 — Herbaceous Fallow (Jachere Herbacee): abandoned field dominated by grasses/weeds, <10% woody cover
# - 32 — Shrubby Fallow (Jachere Arbustive): fallow with regrowth of shrubs (<5 m)
# - 33 — Tree Fallow (Jachere Arboree): fallow with regenerating trees (>5 m)

SYSTEM_PROMPT = """You are an expert in land cover and land use classification for West African landscapes (Benin/Sahel region). You will be shown a ground-level street view photograph. Your task is to classify the dominant land cover visible in the image into EXACTLY ONE of the classes listed below.

## Classification Classes

### Crops (Annual)
- 11 — Maize (Mais): tall grass-like plants with broad leaves, cobs, planted in rows
- 12 — Sorghum/Millet (Sorgho/Mil): tall cereals with grain heads/panicles at the top
- 13 — Rice (Riz): grown in flooded paddies or wet lowlands, dense green grass-like plants
- 14 — Yam (Igname): mounded soil (buttes), climbing vines often on stakes
- 15 — Cassava (Manioc): shrubby plant, palmate leaves, 1–3 m tall
- 16 — Cotton (Coton): bushy plants, white bolls when mature
- 17 — Peanut/Groundnut (Arachide): low-growing legume, small yellow flowers
- 18 — Cowpea (Niebe): legume with trifoliate leaves, often intercropped
- 19 — Soybean (Soja): bushy legume with trifoliate leaves
- 91 — Bambara groundnut (Voandzou): low ground-hugging legume
- 20 — Other Crop (Autre Culture): any cultivated field not matching above

### Orchards / Tree Crops
- 21 — Cashew orchard (Anacardier): evenly spaced trees, rounded dense canopy, often with fruit
- 22 — Mango orchard (Manguier): large dome-shaped evergreen trees, dark dense foliage
- 23 — Teak plantation (Teck): tall straight trees, very large broad leaves, planted in rows
- 29 — Other Orchard (Autre Verger): planted tree crops not matching above



### Anthropogenic / Other
- 51 — Built-up / Artificial Surface (Surface Artificialisee): buildings, roads, paved areas, villages
- 61 — Water (Eau): rivers, ponds, lakes, reservoirs
- 99 — Other Class (Autre Classe): does not fit any class above, or image unusable

## Decision Rules
1. Classify the **dominant** land cover occupying the largest share of the scene (excluding the road/foreground if it is a street view artifact).
2. Distinguish **fallow (31–33)** from **savanna (41–43)**: fallow shows signs of past cultivation (old ridges, crop residues, field boundaries, proximity to active fields); savanna appears natural with no cultivation signs.
3. Distinguish **orchard (21–29)** from **tree savanna (43)**: orchards have regular spacing and planted rows; savanna trees are irregularly distributed.
4. If several classes are roughly equal, prefer the class describing the most structurally dominant vegetation layer.
5. If the image is blurry, too dark, or shows only sky/road, return `99`.

## Output Format
Return ONLY a valid JSON object with this exact schema, nothing else:

{
  "class_code": <int>,
  "class_name_en": "<english name>",
  "class_name_fr": "<french name>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<1–3 sentences describing the key visual cues you used>",
  "alternative_code": <int or null>
}"""


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()


def classify_image(image_path: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": "Now classify this image."},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


if __name__ == "__main__":
    raw = classify_image(IMAGE_PATH)
    print("=== Raw model output ===")
    print(raw)

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start : end + 1])
        print("\n=== Parsed JSON ===")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"\n[warn] Could not parse JSON: {exc}")
