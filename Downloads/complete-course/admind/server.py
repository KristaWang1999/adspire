import os, io, json, base64, random, time
import urllib.parse
import concurrent.futures
from collections import defaultdict
import httpx
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
try:
    from openai import OpenAI as _OpenAI
    openai_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "")
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
from PIL import Image, ImageFilter

load_dotenv()
app = Flask(__name__)
CORS(app)

# ── Rate limiting ──
PER_IP_DAILY_LIMIT  = 3    # max generations per IP per day
GLOBAL_DAILY_LIMIT  = 40   # max total generations per day (all users)

_ip_usage    = defaultdict(list)   # ip -> [timestamp, ...]
_global_usage = []                 # [timestamp, ...]
_rl_lock = __import__('threading').Lock()

def _prune(ts_list):
    cutoff = time.time() - 86400
    return [t for t in ts_list if t > cutoff]

WHITELIST_IPS = {'127.0.0.1', '::1'}  # localhost always unrestricted

def check_rate_limit(ip):
    if ip in WHITELIST_IPS:
        return True, None
    with _rl_lock:
        _ip_usage[ip]   = _prune(_ip_usage[ip])
        global _global_usage
        _global_usage   = _prune(_global_usage)
        ip_count     = len(_ip_usage[ip])
        global_count = len(_global_usage)
        if global_count >= GLOBAL_DAILY_LIMIT:
            return False, f"今日体验名额已用完，请明天再试 😊"
        if ip_count >= PER_IP_DAILY_LIMIT:
            return False, f"每位访客每天最多体验 {PER_IP_DAILY_LIMIT} 次，明天见！"
        now = time.time()
        _ip_usage[ip].append(now)
        _global_usage.append(now)
        return True, None

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Re-init openai after load_dotenv so key is available
try:
    from openai import OpenAI as _OpenAI2
    openai_client = _OpenAI2(api_key=os.getenv("OPENAI_API_KEY") or "")
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_AVAILABLE = False

# ── Platform dimensions (fallback / resize target) ──
PLATFORM_DIMS = {
    'instagram': (1024, 1024),
    'xhs':       (1024, 1365),
    'facebook':  (1024, 1280),
    'tiktok':    (1024, 1820),
}

PLATFORM_OAI_SIZE = {
    'instagram': '1024x1024',
    'xhs':       '1024x1536',
    'facebook':  '1024x1536',
    'tiktok':    '1024x1536',
}

# ── Category-aware scene prompts for gpt-image-1 ──
CATEGORY_SCENES = {
    'beauty': [
        'Complete luxury beauty advertisement photo: {product} dramatically lit on polished black marble, one focused overhead spotlight, cinematic deep shadows, dark moody editorial mood.',
        'Premium beauty campaign photo: {product} on pristine white marble, soft diffused studio light, airy minimal composition, high-end skincare brand aesthetic.',
        'Luxury fragrance advertisement: {product} surrounded by pink rose petals and blush botanicals on satin, warm romantic glowing light, editorial beauty photography.',
    ],
    'tech': [
        'Premium tech advertisement photo: {product} on dark polished surface with dramatic neon rim lighting, futuristic cinematic mood, deep blacks.',
        'Minimalist tech product photo: {product} on clean white surface, crisp geometric light and shadow, contemporary premium brand aesthetic.',
        'Bold tech editorial: {product} with electric blue and purple accent lighting on very dark background, dramatic high-tech advertisement style.',
    ],
    'food': [
        'Artisan food advertisement: {product} on warm wood surface, golden hour side lighting, inviting and premium atmosphere.',
        'Gourmet editorial photo: {product} on white marble with complementary ingredients, clean overhead composition, luxury food brand aesthetic.',
        'Premium food lifestyle photo: {product} with warm ambient light, steam, rich and inviting premium food photography.',
    ],
    'fashion': [
        'Fashion editorial flat-lay: {product} on neutral linen texture, soft even studio light, fashion magazine aesthetic.',
        'Luxury fashion campaign: {product} in aspirational lifestyle setting, warm natural light, high-end brand photography.',
        'High-contrast fashion editorial: {product} with dramatic spotlight, bold luxury fashion advertisement photography.',
    ],
    'wellness': [
        'Organic wellness advertisement: {product} on wood with fresh botanical accents, bright morning window light, clean lifestyle aesthetic.',
        'Premium wellness brand photo: {product} in minimal spa-like setting, neutral tones, serene and pure mood.',
        'Natural wellness editorial: {product} with organic ingredients, warm light, clean organic lifestyle photography.',
    ],
    'home': [
        'Aspirational home lifestyle photo: {product} in cozy styled interior, warm ambient light, premium home brand aesthetic.',
        'Minimalist home goods photo: {product} on clean white surface with Scandinavian props, crisp studio light, premium aesthetic.',
        'Luxury interior editorial: {product} in stylish living space, high-end home brand photography.',
    ],
}

DEFAULT_CATEGORY_SCENES = [
    'Premium advertising photo: {product} on clean white marble, soft professional studio lighting, ultra high quality.',
    'Luxury editorial photo: {product} on warm lifestyle surface with elegant side lighting, aspirational brand aesthetic.',
    'Dramatic luxury advertisement: {product} in dark studio with beautiful ambient lighting, cinematic quality.',
]

MOOD_SCENES = {
    '玫瑰粉': [
        'soft pink rose petals scattered on white marble, warm bokeh, empty surface, no products, editorial background',
        'blush satin fabric with dried pink flowers, romantic soft focus, clean empty background',
        'peach botanicals and cream linen texture, flat lay background, empty, no objects',
    ],
    '深邃黑': [
        'black polished marble with dramatic side lighting, luxury empty background, no products',
        'dark moody studio with single spotlight beam, deep shadows, empty surface, no products',
        'deep black velvet texture with warm gold accent rim light, empty elegant background',
    ],
    '活力橙': [
        'warm terracotta surface in golden hour sunlight, vibrant empty background, no products',
        'tropical citrus peel texture on bright orange fabric, warm lifestyle background, empty',
        'burnt orange linen with warm ambient light, cozy texture, empty background',
    ],
    '清透白': [
        'clean white glass surface with soft diffused studio light, minimal empty background',
        'crystal clear water droplets on white marble, fresh airy minimal empty background',
        'white ceramic tiles with crisp professional lighting, pure empty background',
    ],
    '奶油米': [
        'warm cream linen on natural wood with morning light, beige aesthetic empty background',
        'beige sand texture with warm sunlight soft shadows, empty lifestyle background',
        'ivory marble with dried botanicals, warm editorial empty background',
    ],
    '薰衣草': [
        'purple lavender field blurred bokeh, dreamy purple empty background',
        'lilac silk fabric with soft purple ambient light, elegant empty background',
        'violet abstract gradient studio light, ethereal soft empty background',
    ],
    '湖水蓝': [
        'aqua turquoise water surface with gentle ripples, tropical fresh empty background',
        'teal blue glass surface with ocean-inspired soft light, minimal empty background',
        'blue gradient studio with cool crisp elegant lighting, empty background',
    ],
    '森林绿': [
        'moss and fern on wooden surface, dappled forest light, nature empty background',
        'sage green linen with botanical herbs, flat lay empty background',
        'dark green marble texture, sophisticated minimal empty background',
    ],
}

DEFAULT_SCENES = [
    'clean minimal white marble surface, soft studio lighting, professional empty background',
    'warm natural light with botanical elements on cream surface, lifestyle empty background',
    'dramatic dark studio background with elegant side lighting, luxury empty background',
]

MOOD_GRADIENT = {
    '玫瑰粉': ((253, 232, 240), (195,  95, 135)),
    '深邃黑': (( 15,  15,  22), ( 38,  38,  55)),
    '活力橙': ((255, 245, 225), (215,  95,  25)),
    '清透白': ((238, 248, 255), (185, 215, 240)),
    '奶油米': ((250, 242, 224), (195, 170, 120)),
    '薰衣草': ((240, 235, 255), (150, 125, 210)),
    '湖水蓝': ((225, 245, 255), ( 45, 135, 180)),
    '森林绿': ((224, 245, 232), ( 35, 115,  70)),
}
DEFAULT_GRADIENT = ((235, 235, 245), (100, 115, 160))

def get_scenes(mood=''):
    for k, v in MOOD_SCENES.items():
        if k in mood:
            return v
    return DEFAULT_SCENES

PRESET_STYLES = {
    '暗黑奢华': 'dramatic dark luxury editorial, deep shadows, rich dark background, gold accent rim lighting, mysterious opulent atmosphere, high-end luxury brand',
    '奶油暖调': 'warm cream beige lifestyle, soft natural diffused light, cozy elegant, warm tones, artisan editorial aesthetic',
    '科技极简': 'clean minimal tech aesthetic, dark background, precision geometric lighting, modern sophisticated, premium tech brand',
    '自然清新': 'natural green organic, fresh botanical elements, soft morning window light, clean wellness brand aesthetic',
    '玫瑰浪漫': 'romantic rose pink, soft feminine, delicate floral elements, warm glowing light, luxurious romantic atmosphere',
    '黑白经典': 'classic black and white high contrast, timeless elegant, fine art photography, monochrome luxury editorial',
}

def analyze_style_reference(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe the visual style of this advertisement image for use as a style reference prompt. Cover: color palette, lighting, mood/atmosphere, composition, background aesthetic. 2-3 sentences in English only."},
        ]}],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

def get_category_scenes(category=''):
    cat = (category or '').lower()
    for k in CATEGORY_SCENES:
        if k in cat:
            return CATEGORY_SCENES[k]
    return DEFAULT_CATEGORY_SCENES

def analyze_product(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": 'Analyze this product image. Return JSON only, no extra text:\n{"product_name":"product name in English","headline":"ad headline in Chinese under 12 chars","tagline":"ad subtitle in Chinese under 18 chars","key_features":["feature1","feature2","feature3"],"color_mood":"one of: 玫瑰粉/深邃黑/活力橙/清透白/奶油米/薰衣草/湖水蓝/森林绿","product_category":"one of: beauty/tech/food/fashion/wellness/home/other"}'}
        ]}],
        max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    return json.loads(text[text.find("{"):text.rfind("}")+1])

def generate_ad_copies(product_info):
    product_name = product_info.get('product_name', 'product')
    features = product_info.get('key_features', [])
    headline = product_info.get('headline', '')
    features_str = '、'.join(features) if features else '无'
    prompt = f"""你是专业的中国社交媒体广告文案策划师。

产品：{product_name}
广告语：{headline}
产品特点：{features_str}

为该产品生成15条社媒广告文案，分5种类型各3条：
1. 痛点型（子标签：痛点共鸣，平台：Meta版/小红书版/抖音版）
2. 故事型（子标签：情绪投射，平台：小红书版/抖音版/微信版）
3. 对比型（子标签：逻辑说服，平台：Meta版/小红书版/抖音版）
4. 权威型（子标签：信任背书，平台：Meta版/小红书版/微信版）
5. 悬念型（子标签：好奇驱动，平台：抖音版/小红书版/Meta版）

要求：紧扣产品特点，不要说泛泛话，文案自然接地气有吸引力。
只返回JSON，格式：{{"copies":[{{"type":"痛点型","subtype":"痛点共鸣","platform":"Meta版","text":"..."}}]}}"""
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw[raw.find("{"):raw.rfind("}")+1]).get('copies', [])
    except Exception as e:
        print(f"Copy generation failed: {e}")
        return []

def generate_ad_strategy(product_info):
    product_name = product_info.get('product_name', 'product')
    features = product_info.get('key_features', [])
    category = product_info.get('product_category', 'other')
    features_str = '、'.join(features) if features else '无'
    prompt = f"""你是专业的中国社媒广告投放策略师。

产品：{product_name}
类别：{category}
特点：{features_str}

请生成针对该产品的投放建议。只返回JSON，格式如下：
{{
  "meta": {{
    "audience": "目标受众描述（年龄/性别/兴趣）",
    "lookalike": "Lookalike受众建议",
    "test": "A/B测试建议"
  }},
  "xiaohongshu": {{
    "format": "内容形式建议",
    "keywords": ["关键词1","关键词2","关键词3"],
    "tip": "运营小技巧"
  }},
  "kpi": {{
    "ctr": "预期CTR范围",
    "cpc": "预期CPC范围（人民币）",
    "roas": "目标ROAS"
  }}
}}"""
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception as e:
        print(f"Strategy generation failed: {e}")
        return {}

def build_ad_prompt(product_name, scene_template, color_mood='', key_features=None,
                    headline='', tagline='', brand_label='', reference_style=''):
    mood_palette = {
        '玫瑰粉': 'soft rose pink and blush tones',
        '深邃黑': 'deep black and dramatic dark tones',
        '活力橙': 'warm vibrant orange and golden tones',
        '清透白': 'clean white and crystal clear tones',
        '奶油米': 'warm cream and beige tones',
        '薰衣草': 'soft purple and lavender tones',
        '湖水蓝': 'aqua blue and teal tones',
        '森林绿': 'deep green and earthy natural tones',
    }

    if reference_style:
        # Reference style mode: style description IS the scene; skip color mood to avoid conflict
        scene = (
            f"{reference_style}. "
            f"The hero product is {product_name}, elegantly placed and sharply in focus at the center of the composition. "
            f"Maintain every aspect of the visual style above — lighting, color palette, textures, mood — exactly as described."
        )
        palette_str = ''
    else:
        scene = scene_template.replace('{product}', product_name)
        palette = ''
        for k, v in mood_palette.items():
            if k in color_mood:
                palette = v
                break
        palette_str = f" Overall color palette: {palette}." if palette else ''

    features_str = f" Key features: {', '.join(key_features[:2])}." if key_features else ''
    text_str = ''
    if headline:
        brand_str = f'brand name "{brand_label.upper()}" in elegant small caps, ' if brand_label else ''
        text_str = (
            f' This is a complete finished advertisement — integrate elegant typographic text directly into the image composition: '
            f'{brand_str}'
            f'large bold Chinese headline text "{headline}", '
            f'smaller subtitle "{tagline}" below it. '
            f'Use luxury fashion magazine typography. White or off-white lettering. '
            f'Compose the image and text as a single cohesive design, professionally laid out.'
        )
    return f"{scene}{palette_str}{features_str}{text_str} 4K ultra high quality professional advertising photography."

def generate_ad_openai(product_name, scene_template, color_mood='', key_features=None,
                       platform_key='instagram', headline='', tagline='', brand_label='',
                       reference_style=''):
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        return None
    size = PLATFORM_OAI_SIZE.get(platform_key, '1024x1024')
    prompt = build_ad_prompt(product_name, scene_template, color_mood, key_features,
                             headline, tagline, brand_label, reference_style)
    print(f"  OpenAI prompt ({len(prompt)} chars): {prompt[:100]}...")
    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size,
            quality="low",
        )
        b64 = response.data[0].b64_json
        if b64:
            return base64.b64decode(b64)
    except Exception as e:
        print(f"OpenAI generation failed: {e}")
    return None

def remove_bg_rgba(image_bytes):
    if not REMBG_AVAILABLE:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    out = rembg_remove(image_bytes)
    img = Image.open(io.BytesIO(out)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_bg_pollinations(scene, width, height):
    prompt = f"{scene}, ultra high quality, 8k"
    encoded = urllib.parse.quote(prompt)
    for attempt in range(2):
        seed = random.randint(10000, 99999)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&nologo=true&seed={seed}&model=flux&enhance=false"
        )
        try:
            resp = httpx.get(url, timeout=50, follow_redirects=True)
            resp.raise_for_status()
            if len(resp.content) > 5000:
                return resp.content
        except Exception as e:
            print(f"Pollinations attempt {attempt+1} failed: {e}")
            if attempt == 0:
                time.sleep(2)
    return None

def generate_gradient_bg(mood, width, height):
    c1, c2 = DEFAULT_GRADIENT
    for k, v in MOOD_GRADIENT.items():
        if k in mood:
            c1, c2 = v
            break
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    t = (xx * 0.35 + yy * 0.65).clip(0, 1)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        arr[:, :, i] = (c1[i] + (c2[i] - c1[i]) * t).astype(np.uint8)
    cx, cy = width / 2, height * 0.4
    dist = np.sqrt(((xx * width - cx) / width)**2 + ((yy * height - cy) / height)**2)
    highlight = (1 - np.clip(dist * 1.4, 0, 1)) * 18
    arr = np.clip(arr.astype(np.int16) + highlight[:, :, np.newaxis].astype(np.int16), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return buf.getvalue()

def composite_product(bg_bytes, product_rgba_bytes, width, height):
    bg = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
    bg = bg.resize((width, height), Image.LANCZOS)
    product = Image.open(io.BytesIO(product_rgba_bytes)).convert("RGBA")
    short = min(width, height)
    target_w = int(short * 0.63)
    ratio = target_w / product.width
    target_h = int(product.height * ratio)
    if target_h > height * 0.80:
        target_h = int(height * 0.80)
        ratio = target_h / product.height
        target_w = int(product.width * ratio)
    product = product.resize((target_w, target_h), Image.LANCZOS)
    _, _, _, alpha = product.split()
    shadow_alpha = alpha.point(lambda v: min(v, 130))
    shadow = Image.new("RGBA", (target_w + 60, target_h + 60), (0, 0, 0, 0))
    shadow_body = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))
    shadow_body.putalpha(shadow_alpha)
    shadow.paste(shadow_body, (30, 30), shadow_body)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    x = (width - target_w) // 2
    y = height - target_h - int(height * 0.06)
    canvas = bg.copy()
    canvas.paste(shadow, (x - 30, y - 30), shadow)
    canvas.paste(product, (x, y), product)
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()

def resize_to_platform(img_bytes, width, height):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def index():
    return send_from_directory('.', 'prototype.html')

@app.route("/api/generate", methods=["POST"])
def generate():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    allowed, msg = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": msg, "rate_limited": True}), 429

    if "image" not in request.files:
        return jsonify({"error": "请上传产品图片"}), 400
    image_bytes = request.files["image"].read()

    try:
        copy = analyze_product(image_bytes)
    except Exception as e:
        return jsonify({"error": f"产品识别失败：{e}"}), 500

    try:
        product_rgba_bytes = remove_bg_rgba(image_bytes)
        product_b64 = base64.b64encode(product_rgba_bytes).decode()
    except Exception as e:
        print(f"BG removal failed: {e}")
        product_rgba_bytes = None
        product_b64 = None

    mood       = copy.get("color_mood", "")
    category   = copy.get("product_category", "other")
    product_nm = copy.get("product_name", "product")
    features   = copy.get("key_features", [])
    headline   = copy.get("headline", "")
    tagline    = copy.get("tagline", "")
    platform_key    = request.form.get("platform", "instagram")
    reference_style = request.form.get("reference_style", "")
    bg_w, bg_h      = PLATFORM_DIMS.get(platform_key, (1024, 1024))

    cat_scenes  = get_category_scenes(category)
    mood_scenes = get_scenes(mood)

    print(f"Product: {product_nm} | Category: {category} | Mood: {mood} | OpenAI: {OPENAI_AVAILABLE}")

    def gen_one(i):
        # ── Try OpenAI gpt-image-1 first ──
        if OPENAI_AVAILABLE:
            print(f"  [gen {i}] Trying OpenAI...")
            result = generate_ad_openai(
                product_nm,
                cat_scenes[i % len(cat_scenes)],
                color_mood=mood,
                key_features=features,
                platform_key=platform_key,
                headline=headline,
                tagline=tagline,
                brand_label=product_nm,
                reference_style=reference_style,
            )
            if result and len(result) > 5000:
                print(f"  [gen {i}] OpenAI success ({len(result)//1024}KB)")
                return resize_to_platform(result, bg_w, bg_h)
            print(f"  [gen {i}] OpenAI failed, falling back to Pollinations")

        # ── Fallback: Pollinations + composite ──
        time.sleep(i * 1.5)
        bg_bytes = generate_bg_pollinations(mood_scenes[i % len(mood_scenes)], bg_w, bg_h)
        if bg_bytes is None:
            print(f"  [gen {i}] Pollinations failed, using gradient")
            bg_bytes = generate_gradient_bg(mood, bg_w, bg_h)
        try:
            if product_rgba_bytes:
                return composite_product(bg_bytes, product_rgba_bytes, bg_w, bg_h)
            return base64.b64encode(bg_bytes).decode()
        except Exception as e:
            print(f"  Composite {i} failed: {e}")
            return base64.b64encode(bg_bytes).decode()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        img_futures = [ex.submit(gen_one, i) for i in range(2)]
        copy_future     = ex.submit(generate_ad_copies, copy)
        strategy_future = ex.submit(generate_ad_strategy, copy)
        generated   = [f.result() for f in img_futures]
        ad_copies   = copy_future.result()
        ad_strategy = strategy_future.result()

    return jsonify({
        "ad_copy": copy,
        "product_b64": product_b64,
        "generated": generated,
        "ad_copies": ad_copies,
        "ad_strategy": ad_strategy,
    })

@app.route("/api/style-transfer", methods=["POST"])
def style_transfer():
    if "product" not in request.files:
        return jsonify({"error": "请上传产品图"}), 400
    product_bytes = request.files["product"].read()

    # Get style description
    style_name = request.form.get("style_name", "")
    ref_file = request.files.get("reference")

    if ref_file:
        ref_bytes = ref_file.read()
        style_desc = analyze_style_reference(ref_bytes)
        print(f"Style from reference: {style_desc[:80]}...")
    elif style_name in PRESET_STYLES:
        style_desc = PRESET_STYLES[style_name]
        print(f"Style preset: {style_name}")
    else:
        style_desc = "premium professional advertising photography"

    # Analyze product for name
    try:
        info = analyze_product(product_bytes)
        product_name = info.get("product_name", "product")
    except Exception:
        product_name = "product"

    prompt = (
        f"Professional advertising photograph of {product_name}. "
        f"Visual style: {style_desc}. "
        f"The product is the hero subject, beautifully lit and composed in this style. "
        f"No text or watermarks. Ultra high quality."
    )
    print(f"Style transfer prompt: {prompt[:120]}...")

    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI 未配置"}), 500
    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="low",
        )
        b64 = response.data[0].b64_json
        return jsonify({"image": b64, "style_used": style_desc[:80]})
    except Exception as e:
        print(f"Style transfer failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/feed")
def feed():
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'ads.db')
    if not os.path.exists(db_path):
        return jsonify({"ads": [], "total": 0})
    category = request.args.get('category', '')
    limit = int(request.args.get('limit', 20))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if category and category != '全部':
        rows = conn.execute(
            'SELECT * FROM ads WHERE category=? ORDER BY fetched_at DESC LIMIT ?',
            (category, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM ads ORDER BY fetched_at DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return jsonify({"ads": [dict(r) for r in rows], "total": len(rows)})

@app.route("/ad/<path:filename>")
def serve_ad(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'ad'), filename)

@app.route("/api/discover")
def discover():
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'ads.db')
    if not os.path.exists(db_path):
        return jsonify({})
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    categories = ['护肤', '科技', '时尚', '健康', '食品']
    result = {}
    all_ads = []

    # Build base URL from current request so images resolve on any host
    base = request.host_url.rstrip('/')

    def fix_url(url):
        if not url:
            return url
        # Replace any hardcoded localhost origin with current host
        import re
        return re.sub(r'https?://[^/]+(/ad/)', base + r'\1', url)

    for cat in categories:
        rows = conn.execute(
            'SELECT * FROM ads WHERE category=? ORDER BY RANDOM() LIMIT 8', (cat,)
        ).fetchall()
        ads = []
        for r in rows:
            d = dict(r)
            d['image_url'] = fix_url(d.get('image_url'))
            d['thumb_url'] = fix_url(d.get('thumb_url'))
            ads.append(d)
        if ads:
            result[cat] = ads
            all_ads.extend(ads[:2])
    if all_ads:
        random.shuffle(all_ads)
        result['全部'] = all_ads
    conn.close()
    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
