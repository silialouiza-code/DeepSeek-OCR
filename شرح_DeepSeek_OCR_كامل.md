# شرح شامل لمشروع DeepSeek-OCR مع الأمثلة

## 📋 جدول المحتويات
1. [نظرة عامة على المشروع](#نظرة-عامة)
2. [ما هو DeepSeek-OCR؟](#ما-هو-deepseek-ocr)
3. [المميزات الرئيسية](#المميزات-الرئيسية)
4. [البنية التقنية](#البنية-التقنية)
5. [التثبيت والإعداد](#التثبيت-والإعداد)
6. [أوضاع التشغيل](#أوضاع-التشغيل)
7. [أمثلة عملية مفصلة](#أمثلة-عملية-مفصلة)
8. [شرح الأكواد](#شرح-الأكواد)
9. [حالات الاستخدام](#حالات-الاستخدام)

---

## 🎯 نظرة عامة

**DeepSeek-OCR** هو نموذج ذكاء اصطناعي متقدم تم تطويره بواسطة DeepSeek AI لاستخراج النصوص من الصور والمستندات (OCR - Optical Character Recognition) مع ضغط بصري للسياق.

### معلومات المشروع:
- **المطور**: DeepSeek AI
- **النوع**: نموذج Vision-Language Model (VLM)
- **الهدف**: استخراج النصوص وتحويل المستندات إلى Markdown
- **الترخيص**: مفتوح المصدر
- **تاريخ الإصدار**: أكتوبر 2025

---

## 🤔 ما هو DeepSeek-OCR؟

DeepSeek-OCR هو نموذج يجمع بين:
1. **التعرف الضوئي على الحروف (OCR)**: قراءة النصوص من الصور
2. **فهم التخطيط (Layout Understanding)**: فهم بنية المستند
3. **التحويل الذكي**: تحويل المستندات إلى Markdown مع الحفاظ على التنسيق
4. **الضغط البصري**: تقليل عدد الرموز البصرية (Vision Tokens) مع الحفاظ على الدقة

### الفرق بين DeepSeek-OCR والـ OCR التقليدي:

| الميزة | OCR التقليدي | DeepSeek-OCR |
|--------|--------------|--------------|
| استخراج النص | ✅ نعم | ✅ نعم |
| فهم التخطيط | ❌ محدود | ✅ متقدم |
| التحويل إلى Markdown | ❌ لا | ✅ نعم |
| التعرف على الجداول | ⚠️ ضعيف | ✅ ممتاز |
| التعرف على الصيغ الرياضية | ❌ لا | ✅ نعم |
| استخراج الصور من المستندات | ❌ لا | ✅ نعم |
| عدد الرموز البصرية | 🔴 عالي | 🟢 منخفض (64-400) |

---

## ⭐ المميزات الرئيسية

### 1. **أوضاع دقة متعددة**
يدعم النموذج 5 أوضاع مختلفة حسب حجم الصورة:

```python
# Tiny: للصور الصغيرة - 64 رمز بصري فقط
base_size = 512, image_size = 512, crop_mode = False

# Small: للصور المتوسطة - 100 رمز بصري
base_size = 640, image_size = 640, crop_mode = False

# Base: للصور القياسية - 256 رمز بصري
base_size = 1024, image_size = 1024, crop_mode = False

# Large: للصور الكبيرة - 400 رمز بصري
base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: للصور الديناميكية - n×640×640 + 1×1024×1024
base_size = 1024, image_size = 640, crop_mode = True
```

### 2. **أنواع المخرجات**
- **Markdown**: تحويل المستند إلى Markdown كامل
- **Free OCR**: استخراج النص فقط بدون تنسيق
- **Grounding**: استخراج النص مع تحديد المواقع
- **وصف الصور**: وصف تفصيلي للصور

### 3. **معالجة ملفات PDF**
- تحويل PDF متعدد الصفحات إلى Markdown
- معالجة متوازية للصفحات
- سرعة عالية: ~2500 رمز/ثانية على A100-40G

---

## 🏗️ البنية التقنية

### مكونات المشروع:

```
DeepSeek-OCR/
├── DeepSeek-OCR-master/
│   ├── DeepSeek-OCR-hf/          # استخدام مكتبة Transformers
│   │   └── run_dpsk_ocr.py       # سكريبت التشغيل الأساسي
│   └── DeepSeek-OCR-vllm/        # استخدام مكتبة vLLM (أسرع)
│       ├── config.py             # ملف الإعدادات
│       ├── deepseek_ocr.py       # النموذج الأساسي
│       ├── run_dpsk_ocr_image.py # معالجة الصور
│       ├── run_dpsk_ocr_pdf.py   # معالجة PDF
│       └── run_dpsk_ocr_eval_batch.py # التقييم الدفعي
├── requirements.txt              # المكتبات المطلوبة
└── README.md                     # التوثيق
```

### المكتبات المستخدمة:

```txt
transformers==4.46.3    # للتعامل مع النماذج
tokenizers==0.20.3      # لتقسيم النصوص
PyMuPDF                 # لقراءة PDF
img2pdf                 # لتحويل الصور إلى PDF
einops                  # لعمليات Tensor
easydict                # للقواميس السهلة
addict                  # للقواميس المتقدمة
Pillow                  # لمعالجة الصور
numpy                   # للعمليات الرياضية
```

---

## 🔧 التثبيت والإعداد

### الطريقة 1: باستخدام Transformers (أبسط)

```bash
# 1. إنشاء بيئة افتراضية
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# 2. تثبيت المكتبات
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation

# 3. تشغيل النموذج
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

### الطريقة 2: باستخدام vLLM (أسرع)

```bash
# 1. تثبيت vLLM
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 2. تعديل الإعدادات في config.py
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
nano config.py  # أو أي محرر نصوص

# 3. تشغيل النموذج
python run_dpsk_ocr_image.py  # للصور
python run_dpsk_ocr_pdf.py    # لملفات PDF
```

---

## 🎮 أوضاع التشغيل

### 1. **وضع Transformers** (للاستخدام البسيط)

#### مثال: استخراج نص من صورة

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

# إعداد البيئة
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# تحميل النموذج
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# الإعدادات
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'path/to/your/image.jpg'
output_path = 'output/directory'

# التشغيل
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,      # حجم الصورة الأساسي
    image_size=640,      # حجم القطع
    crop_mode=True,      # تفعيل وضع القص
    save_results=True,   # حفظ النتائج
    test_compress=True   # اختبار الضغط
)

print(result)
```

### 2. **وضع vLLM** (للأداء العالي)

#### مثال: معالجة صورة واحدة مع Streaming

```python
import asyncio
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from PIL import Image

async def process_image(image_path, prompt):
    # إعداد المحرك
    engine_args = AsyncEngineArgs(
        model='deepseek-ai/DeepSeek-OCR',
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # إعدادات التوليد
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
    )
    
    # تحميل الصورة
    image = Image.open(image_path).convert('RGB')
    
    # إنشاء الطلب
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }
    
    # التوليد مع Streaming
    request_id = "request-1"
    full_text = ""
    
    async for output in engine.generate(request, sampling_params, request_id):
        if output.outputs:
            full_text = output.outputs[0].text
            print(full_text[-50:], end='', flush=True)  # طباعة آخر 50 حرف
    
    return full_text

# التشغيل
result = asyncio.run(process_image(
    'document.jpg',
    '<image>\n<|grounding|>Convert the document to markdown.'
))
```

#### مثال: معالجة ملف PDF كامل

```python
import fitz  # PyMuPDF
from PIL import Image
from vllm import LLM, SamplingParams
import io

def pdf_to_images(pdf_path, dpi=144):
    """تحويل PDF إلى صور"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    pdf_document.close()
    return images

# تحميل النموذج
llm = LLM(
    model='deepseek-ai/DeepSeek-OCR',
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    max_model_len=8192,
    max_num_seqs=100,  # عدد الطلبات المتزامنة
    gpu_memory_utilization=0.9,
)

# إعدادات التوليد
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# تحويل PDF إلى صور
pdf_path = 'document.pdf'
images = pdf_to_images(pdf_path)

# إنشاء طلبات دفعية
batch_inputs = []
prompt = '<image>\n<|grounding|>Convert the document to markdown.'

for image in images:
    batch_inputs.append({
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    })

# معالجة جميع الصفحات دفعة واحدة
outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

# حفظ النتائج
full_markdown = ""
for i, output in enumerate(outputs):
    page_content = output.outputs[0].text
    full_markdown += f"\n\n--- Page {i+1} ---\n\n{page_content}"

# حفظ في ملف
with open('output.md', 'w', encoding='utf-8') as f:
    f.write(full_markdown)

print(f"تم معالجة {len(images)} صفحة بنجاح!")
```

---

## 📝 أمثلة عملية مفصلة

### مثال 1: استخراج نص من فاتورة

```python
from transformers import AutoModel, AutoTokenizer
import torch

# تحميل النموذج
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# معالجة فاتورة
prompt = "<image>\n<|grounding|>Convert the document to markdown."
invoice_image = 'invoice.jpg'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=invoice_image,
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)

print("محتوى الفاتورة:")
print(result)
```

**النتيجة المتوقعة:**
```markdown
# Invoice

**Invoice Number:** INV-2025-001
**Date:** 2025-01-15

## Bill To:
John Doe
123 Main Street
City, State 12345

## Items:

| Item | Quantity | Price | Total |
|------|----------|-------|-------|
| Product A | 2 | $50.00 | $100.00 |
| Product B | 1 | $75.00 | $75.00 |

**Subtotal:** $175.00
**Tax (10%):** $17.50
**Total:** $192.50
```

### مثال 2: استخراج جدول من صورة

```python
# استخدام وضع Grounding لاستخراج الجداول
prompt = "<image>\n<|grounding|>Extract all tables from this image."
table_image = 'table.png'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=table_image,
    output_path='./tables',
    base_size=1024,
    image_size=1024,
    crop_mode=False,  # بدون قص للجداول
    save_results=True
)

print("الجدول المستخرج:")
print(result)
```

### مثال 3: استخراج معادلات رياضية

```python
# استخراج المعادلات الرياضية من ورقة بحثية
prompt = "<image>\n<|grounding|>Convert the document to markdown."
math_paper = 'research_paper.jpg'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=math_paper,
    output_path='./math_output',
    base_size=1280,  # دقة عالية للمعادلات
    image_size=1280,
    crop_mode=False,
    save_results=True
)

print("المعادلات المستخرجة:")
print(result)
```

**النتيجة المتوقعة:**
```markdown
The quadratic formula is:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

Where:
- $a$, $b$, $c$ are coefficients
- $x$ is the solution
```

### مثال 4: وصف صورة عامة

```python
# وصف صورة بدون OCR
prompt = "<image>\nDescribe this image in detail."
photo = 'landscape.jpg'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=photo,
    output_path='./descriptions',
    base_size=640,
    image_size=640,
    crop_mode=False,
    save_results=True
)

print("وصف الصورة:")
print(result)
```

### مثال 5: استخراج نص فقط (Free OCR)

```python
# استخراج النص بدون تنسيق
prompt = "<image>\nFree OCR."
document = 'simple_text.jpg'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=document,
    output_path='./text_only',
    base_size=512,  # حجم صغير للنص البسيط
    image_size=512,
    crop_mode=False,
    save_results=True
)

print("النص المستخرج:")
print(result)
```

---

## 🔍 شرح الأكواد

### 1. شرح ملف `config.py`

```python
# ملف الإعدادات الرئيسي لـ vLLM

# أوضاع الدقة المختلفة
# Tiny: 512×512 (64 رمز بصري) - للصور الصغيرة
# Small: 640×640 (100 رمز بصري) - للصور المتوسطة
# Base: 1024×1024 (256 رمز بصري) - للصور القياسية
# Large: 1280×1280 (400 رمز بصري) - للصور الكبيرة
# Gundam: n×640×640 + 1×1024×1024 - للصور الديناميكية

BASE_SIZE = 1024        # حجم الصورة الأساسي
IMAGE_SIZE = 640        # حجم القطع
CROP_MODE = True        # تفعيل وضع القص الديناميكي

MIN_CROPS = 2           # الحد الأدنى لعدد القطع
MAX_CROPS = 6           # الحد الأقصى لعدد القطع (9 كحد أقصى)

MAX_CONCURRENCY = 100   # عدد الطلبات المتزامنة
NUM_WORKERS = 64        # عدد العمال لمعالجة الصور

PRINT_NUM_VIS_TOKENS = False  # طباعة عدد الرموز البصرية
SKIP_REPEAT = True            # تخطي التكرارات

MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # مسار النموذج

# مسارات الإدخال والإخراج
INPUT_PATH = ''   # مسار الملف المدخل (صورة أو PDF)
OUTPUT_PATH = ''  # مسار مجلد الإخراج

# الأوامر الشائعة (Prompts)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# أمثلة أخرى:
# PROMPT = '<image>\nFree OCR.'  # نص فقط
# PROMPT = '<image>\n<|grounding|>OCR this image.'  # OCR عام
# PROMPT = '<image>\nParse the figure.'  # تحليل الأشكال
# PROMPT = '<image>\nDescribe this image in detail.'  # وصف الصورة

# تحميل Tokenizer
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
```

### 2. شرح `run_dpsk_ocr_image.py`

```python
# هذا السكريبت لمعالجة صورة واحدة مع Streaming Output

import asyncio
from vllm import AsyncLLMEngine, SamplingParams
from PIL import Image

async def stream_generate(image=None, prompt=''):
    """
    دالة لتوليد النص من الصورة مع عرض النتائج تدريجياً
    
    المعاملات:
    - image: الصورة المعالجة
    - prompt: الأمر النصي
    
    العودة:
    - النص الكامل المستخرج
    """
    
    # إعداد محرك vLLM
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,           # حجم الكتلة
        max_model_len=8192,       # الحد الأقصى للطول
        enforce_eager=False,      # عدم فرض التنفيذ الفوري
        trust_remote_code=True,   # الثقة بالكود المخصص
        tensor_parallel_size=1,   # عدد GPUs
        gpu_memory_utilization=0.75,  # استخدام 75% من ذاكرة GPU
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # معالج منع التكرار
    # يمنع تكرار نفس الكلمات في نافذة معينة
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=30,      # حجم N-gram
            window_size=90,     # حجم النافذة
            whitelist_token_ids={128821, 128822}  # رموز مسموحة: <td>, </td>
        )
    ]
    
    # إعدادات التوليد
    sampling_params = SamplingParams(
        temperature=0.0,        # بدون عشوائية (نتائج محددة)
        max_tokens=8192,        # الحد الأقصى للرموز
        logits_processors=logits_processors,
        skip_special_tokens=False,  # عدم تخطي الرموز الخاصة
    )
    
    # إنشاء الطلب
    request_id = f"request-{int(time.time())}"
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }
    
    # التوليد التدريجي
    printed_length = 0
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)  # طباعة النص الجديد فقط
            printed_length = len(full_text)
            final_output = full_text
    
    print('\n')
    return final_output

# الاستخدام
if __name__ == "__main__":
    # تحميل الصورة
    image = load_image(INPUT_PATH).convert('RGB')
    
    # معالجة الصورة
    if '<image>' in PROMPT:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image],
            bos=True,      # إضافة رمز البداية
            eos=True,      # إضافة رمز النهاية
            cropping=CROP_MODE  # تفعيل القص
        )
    
    # التشغيل
    result = asyncio.run(stream_generate(image_features, PROMPT))
    
    # حفظ النتائج
    with open(f'{OUTPUT_PATH}/result.mmd', 'w', encoding='utf-8') as f:
        f.write(result)
```

### 3. شرح `run_dpsk_ocr_pdf.py`

```python
# هذا السكريبت لمعالجة ملفات PDF متعددة الصفحات

import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams

def pdf_to_images_high_quality(pdf_path, dpi=144):
    """
    تحويل PDF إلى صور عالية الجودة
    
    المعاملات:
    - pdf_path: مسار ملف PDF
    - dpi: دقة الصورة (144 افتراضياً)
    
    العودة:
    - قائمة من الصور (PIL Images)
    """
    images = []
    pdf_document = fitz.open(pdf_path)
    
    # حساب معامل التكبير
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    # تحويل كل صفحة إلى صورة
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        # تحويل إلى PIL Image
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    pdf_document.close()
    return images

def process_single_image(image):
    """
    معالجة صورة واحدة وإنشاء طلب vLLM
    
    المعاملات:
    - image: صورة PIL
    
    العودة:
    - قاموس الطلب
    """
    cache_item = {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        },
    }
    return cache_item

if __name__ == "__main__":
    # تحميل النموذج
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        max_model_len=8192,
        max_num_seqs=MAX_CONCURRENCY,  # عدد الطلبات المتزامنة
        gpu_memory_utilization=0.9,    # استخدام 90% من ذاكرة GPU
        disable_mm_preprocessor_cache=True  # تعطيل التخزين المؤقت
    )
    
    # تحويل PDF إلى صور
    print('تحميل PDF...')
    images = pdf_to_images_high_quality(INPUT_PATH)
    
    # معالجة الصور بالتوازي
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(executor.map(process_single_image, images))
    
    # معالجة جميع الصفحات دفعة واحدة
    print('معالجة الصفحات...')
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
    
    # حفظ النتائج
    full_content = ''
    for i, output in enumerate(outputs_list):
        page_content = output.outputs[0].text
        full_content += f'\n<--- Page {i+1} --->\n{page_content}'
    
    # حفظ في ملف Markdown
    output_file = OUTPUT_PATH + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '.mmd')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f'تم معالجة {len(images)} صفحة بنجاح!')
```

---

## 💡 حالات الاستخدام

### 1. **رقمنة المستندات القديمة**
```python
# تحويل مستندات ورقية قديمة إلى Markdown
prompt = "<image>\n<|grounding|>Convert the document to markdown."
old_document = 'old_paper.jpg'

result = model.infer(tokenizer, prompt=prompt, image_file=old_document,
                     base_size=1280, image_size=1280, crop_mode=False)
```

### 2. **استخراج البيانات من الفواتير**
```python
# استخراج معلومات الفواتير تلقائياً
prompt = "<image>\n<|grounding|>Extract invoice details."
invoice = 'invoice_2025.jpg'

result = model.infer(tokenizer, prompt=prompt, image_file=invoice,
                     base_size=1024, image_size=640, crop_mode=True)

# يمكن بعد ذلك تحليل النتيجة باستخدام regex أو NLP
```

### 3. **تحويل الكتب إلى نصوص رقمية**
```python
# تحويل صفحات كتاب إلى Markdown
book_pdf = 'book.pdf'
images = pdf_to_images_high_quality(book_pdf)

# معالجة كل صفحة
for i, image in enumerate(images):
    result = model.infer(tokenizer, 
                        prompt="<image>\n<|grounding|>Convert to markdown.",
                        image_file=image,
                        output_path=f'./book_output/page_{i+1}')
```

### 4. **استخراج الجداول من التقارير**
```python
# استخراج جداول من تقرير مالي
prompt = "<image>\n<|grounding|>Extract all tables."
financial_report = 'report.jpg'

result = model.infer(tokenizer, prompt=prompt, image_file=financial_report,
                     base_size=1024, image_size=1024, crop_mode=False)

# النتيجة ستكون جداول Markdown يمكن تحويلها إلى CSV
```

### 5. **تحليل الأوراق البحثية**
```python
# استخراج المعادلات والنصوص من ورقة بحثية
prompt = "<image>\n<|grounding|>Convert the document to markdown."
research_paper = 'paper.pdf'

# معالجة الورقة البحثية
result = model.infer(tokenizer, prompt=prompt, image_file=research_paper,
                     base_size=1280, image_size=1280, crop_mode=False)

# النتيجة ستحتوي على المعادلات بصيغة LaTeX
```

---

## 🎯 الأوامر (Prompts) الشائعة

### قائمة الأوامر المدعومة:

```python
# 1. تحويل مستند إلى Markdown (مع التخطيط)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# 2. OCR عام لأي صورة
prompt = "<image>\n<|grounding|>OCR this image."

# 3. استخراج نص فقط (بدون تنسيق)
prompt = "<image>\nFree OCR."

# 4. تحليل الأشكال والرسوم البيانية
prompt = "<image>\nParse the figure."

# 5. وصف تفصيلي للصورة
prompt = "<image>\nDescribe this image in detail."

# 6. تحديد موقع نص معين في الصورة
prompt = "<image>\nLocate <|ref|>النص المطلوب<|/ref|> in the image."

# 7. استخراج جداول فقط
prompt = "<image>\n<|grounding|>Extract all tables from this document."

# 8. استخراج معادلات رياضية
prompt = "<image>\n<|grounding|>Extract all mathematical equations."
```

---

## ⚙️ تحسين الأداء

### نصائح لتحسين السرعة:

```python
# 1. استخدام vLLM بدلاً من Transformers
# vLLM أسرع بـ 5-10 مرات

# 2. زيادة MAX_CONCURRENCY للمعالجة المتوازية
MAX_CONCURRENCY = 100  # للـ GPU القوية

# 3. استخدام وضع Tiny للصور الصغيرة
base_size = 512
image_size = 512
crop_mode = False

# 4. تقليل max_tokens إذا كان النص قصيراً
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,  # بدلاً من 8192
)

# 5. استخدام معالجة دفعية للصور المتعددة
batch_inputs = [process_image(img) for img in images]
outputs = llm.generate(batch_inputs, sampling_params)
```

### نصائح لتحسين الدقة:

```python
# 1. استخدام دقة أعلى للمستندات المعقدة
base_size = 1280
image_size = 1280
crop_mode = False

# 2. تفعيل وضع Gundam للصور الكبيرة
base_size = 1024
image_size = 640
crop_mode = True

# 3. استخدام DPI عالي عند تحويل PDF
images = pdf_to_images_high_quality(pdf_path, dpi=200)

# 4. استخدام الأمر المناسب للمهمة
# للمستندات: <|grounding|>Convert the document to markdown.
# للنصوص البسيطة: Free OCR.
```

---

## 🐛 حل المشاكل الشائعة

### مشكلة 1: نفاد ذاكرة GPU

```python
# الحل 1: تقليل استخدام الذاكرة
gpu_memory_utilization=0.6  # بدلاً من 0.9

# الحل 2: تقليل MAX_CONCURRENCY
MAX_CONCURRENCY = 50  # بدلاً من 100

# الحل 3: تقليل MAX_CROPS
MAX_CROPS = 4  # بدلاً من 6

# الحل 4: استخدام دقة أقل
base_size = 640  # بدلاً من 1024
```

### مشكلة 2: نتائج غير دقيقة

```python
# الحل 1: زيادة الدقة
base_size = 1280
image_size = 1280

# الحل 2: استخدام الأمر الصحيح
prompt = "<image>\n<|grounding|>Convert the document to markdown."
# بدلاً من
# prompt = "<image>\nFree OCR."

# الحل 3: تحسين جودة الصورة المدخلة
# استخدام DPI أعلى أو تحسين الإضاءة
```

### مشكلة 3: بطء المعالجة

```python
# الحل 1: استخدام vLLM بدلاً من Transformers
# vLLM أسرع بكثير

# الحل 2: زيادة المعالجة المتوازية
MAX_CONCURRENCY = 100
NUM_WORKERS = 64

# الحل 3: استخدام دقة أقل للصور البسيطة
base_size = 512
image_size = 512
```

---

## 📊 مقارنة الأداء

### سرعة المعالجة:

| الطريقة | الصور/الثانية | الرموز/الثانية | استخدام الذاكرة |
|---------|---------------|----------------|-----------------|
| Transformers | 0.5-1 | 500-1000 | متوسط |
| vLLM (Single) | 1-2 | 1000-2000 | متوسط |
| vLLM (Batch) | 5-10 | 2000-2500 | عالي |

### دقة الاستخراج:

| نوع المستند | الدقة | الملاحظات |
|-------------|-------|-----------|
| نصوص بسيطة | 98%+ | ممتاز |
| جداول | 95%+ | جيد جداً |
| معادلات رياضية | 90%+ | جيد |
| خطوط يدوية | 70-80% | متوسط |
| صور منخفضة الجودة | 60-70% | يحتاج تحسين |

---

## 🎓 خلاصة

**DeepSeek-OCR** هو نموذج قوي ومتقدم لاستخراج النصوص من الصور والمستندات مع المميزات التالية:

✅ **المميزات:**
- دقة عالية في استخراج النصوص
- دعم متعدد للغات (بما فيها العربية)
- تحويل ذكي إلى Markdown
- معالجة سريعة للملفات الكبيرة
- دعم الجداول والمعادلات الرياضية
- مفتوح المصدر ومجاني

⚠️ **القيود:**
- يحتاج GPU قوية (يفضل A100 أو أفضل)
- استهلاك عالي للذاكرة
- قد يكون بطيئاً مع Transformers
- دقة متوسطة مع الخطوط اليدوية

🚀 **الاستخدام الأمثل:**
- استخدم vLLM للأداء العالي
- اختر الدقة المناسبة للمهمة
- استخدم المعالجة الدفعية للملفات المتعددة
- اختر الأمر (Prompt) المناسب

---

## 📚 مصادر إضافية

- **الموقع الرسمي**: https://www.deepseek.com/
- **Hugging Face**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **الورقة البحثية**: https://arxiv.org/abs/2510.18234
- **GitHub**: https://github.com/deepseek-ai/DeepSeek-OCR
- **Discord**: https://discord.gg/Tc7c45Zzu5

---

## 📝 ملاحظات ختامية

هذا الشرح يغطي جميع جوانب مشروع DeepSeek-OCR من الأساسيات إلى الاستخدام المتقدم. يمكنك البدء بالأمثلة البسيطة ثم الانتقال إلى الاستخدامات المتقدمة حسب احتياجاتك.

**نصيحة أخيرة:** ابدأ بوضع Transformers للتجربة، ثم انتقل إلى vLLM عندما تحتاج أداءً أعلى.

---

**تم إنشاء هذا الشرح بواسطة Blackbox AI**
**التاريخ: نوفمبر 2025**
