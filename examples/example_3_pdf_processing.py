"""
مثال 3: معالجة ملفات PDF متعددة الصفحات
=========================================

هذا المثال يوضح كيفية معالجة ملف PDF كامل وتحويله إلى Markdown
باستخدام vLLM للحصول على أداء عالي.
"""

import os
import io
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# استيراد vLLM
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# ملاحظة: يجب استيراد النموذج المخصص
# from deepseek_ocr import DeepseekOCRForCausalLM
# from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
# from process.image_process import DeepseekOCRProcessor


def pdf_to_images(pdf_path, dpi=144):
    """
    تحويل ملف PDF إلى قائمة من الصور
    
    المعاملات:
    - pdf_path: مسار ملف PDF
    - dpi: دقة الصورة (144 افتراضياً)
    
    العودة:
    - قائمة من صور PIL
    """
    
    print(f"\n📄 تحويل PDF إلى صور...")
    print(f"   الملف: {pdf_path}")
    print(f"   الدقة: {dpi} DPI")
    
    images = []
    
    try:
        # فتح ملف PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        
        print(f"   عدد الصفحات: {total_pages}")
        
        # حساب معامل التكبير
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        # تحويل كل صفحة
        for page_num in tqdm(range(total_pages), desc="تحويل الصفحات"):
            page = pdf_document[page_num]
            
            # تحويل الصفحة إلى صورة
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # تحويل إلى PIL Image
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # تحويل إلى RGB إذا لزم الأمر
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            
            images.append(img)
        
        pdf_document.close()
        print(f"✓ تم تحويل {len(images)} صفحة")
        
    except Exception as e:
        print(f"✗ خطأ في تحويل PDF: {str(e)}")
        raise
    
    return images


def process_pdf_with_vllm(pdf_path, output_dir='./output/pdf_processing'):
    """
    معالجة ملف PDF باستخدام vLLM
    
    المعاملات:
    - pdf_path: مسار ملف PDF
    - output_dir: مجلد الإخراج
    """
    
    print("=" * 70)
    print("مثال 3: معالجة ملف PDF متعدد الصفحات")
    print("=" * 70)
    
    # الخطوة 1: إنشاء مجلد الإخراج
    print("\n[1/6] إعداد البيئة...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    print(f"✓ مجلد الإخراج: {output_dir}")
    
    # الخطوة 2: تحويل PDF إلى صور
    print("\n[2/6] تحويل PDF إلى صور...")
    images = pdf_to_images(pdf_path, dpi=144)
    
    if not images:
        print("✗ لم يتم العثور على صفحات في PDF")
        return None
    
    # الخطوة 3: تحميل النموذج
    print("\n[3/6] تحميل نموذج vLLM...")
    print("هذا قد يستغرق بضع دقائق...")
    
    # ملاحظة: هذا مثال توضيحي
    # في الاستخدام الفعلي، يجب تسجيل النموذج المخصص
    """
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    
    llm = LLM(
        model='deepseek-ai/DeepSeek-OCR',
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=100,  # عدد الطلبات المتزامنة
        disable_mm_preprocessor_cache=True
    )
    
    print("✓ تم تحميل النموذج")
    
    # الخطوة 4: إعداد معاملات التوليد
    print("\n[4/6] إعداد معاملات التوليد...")
    
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822}  # <td>, </td>
        )
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    print("✓ تم إعداد المعاملات")
    
    # الخطوة 5: معالجة الصور
    print("\n[5/6] معالجة الصفحات...")
    print(f"عدد الصفحات: {len(images)}")
    print("هذا قد يستغرق عدة دقائق...")
    
    # إنشاء طلبات دفعية
    prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    batch_inputs = []
    
    def process_single_image(image):
        return {
            "prompt": prompt,
            "multi_modal_data": {
                "image": DeepseekOCRProcessor().tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=True
                )
            }
        }
    
    # معالجة الصور بالتوازي
    with ThreadPoolExecutor(max_workers=64) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="إعداد الصور"
        ))
    
    # توليد النتائج
    print("\nتوليد النتائج...")
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
    
    print("✓ تمت المعالجة")
    """
    
    # الخطوة 6: حفظ النتائج
    print("\n[6/6] حفظ النتائج...")
    
    # مثال على حفظ النتائج (بدون vLLM الفعلي)
    full_markdown = ""
    
    for i, image in enumerate(images, 1):
        # في الاستخدام الفعلي، استخدم outputs_list[i-1].outputs[0].text
        page_content = f"# صفحة {i}\n\n[محتوى الصفحة {i}]\n\n"
        full_markdown += page_content + "\n<--- فاصل الصفحة --->\n\n"
    
    # حفظ في ملف Markdown
    pdf_name = Path(pdf_path).stem
    output_file = Path(output_dir) / f'{pdf_name}.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    
    print(f"✓ تم حفظ الملف: {output_file}")
    
    # إحصائيات
    print("\n" + "=" * 70)
    print("📊 إحصائيات المعالجة:")
    print("=" * 70)
    print(f"   عدد الصفحات: {len(images)}")
    print(f"   حجم الملف: {len(full_markdown)} حرف")
    print(f"   ملف الإخراج: {output_file}")
    print("=" * 70)
    
    return output_file


def process_pdf_simple(pdf_path, output_dir='./output/pdf_simple'):
    """
    معالجة PDF بطريقة بسيطة باستخدام Transformers
    (أبطأ ولكن أسهل في الاستخدام)
    """
    
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    print("\n" + "=" * 70)
    print("معالجة PDF بطريقة بسيطة (Transformers)")
    print("=" * 70)
    
    # تحميل النموذج
    print("\n[1/4] تحميل النموذج...")
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    print("✓ تم تحميل النموذج")
    
    # تحويل PDF إلى صور
    print("\n[2/4] تحويل PDF...")
    images = pdf_to_images(pdf_path, dpi=144)
    
    # معالجة كل صفحة
    print("\n[3/4] معالجة الصفحات...")
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    full_markdown = ""
    
    for i, image in enumerate(tqdm(images, desc="معالجة الصفحات"), 1):
        # حفظ الصورة مؤقتاً
        temp_image_path = f'{output_dir}/temp_page_{i}.jpg'
        image.save(temp_image_path)
        
        # معالجة الصفحة
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=f'{output_dir}/page_{i}',
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False
            )
            
            full_markdown += f"\n\n# صفحة {i}\n\n{result}\n\n"
            full_markdown += "<--- فاصل الصفحة --->\n\n"
            
        except Exception as e:
            print(f"\n✗ خطأ في معالجة الصفحة {i}: {str(e)}")
            full_markdown += f"\n\n# صفحة {i}\n\n[خطأ في المعالجة]\n\n"
        
        # حذف الصورة المؤقتة
        os.remove(temp_image_path)
    
    # حفظ النتيجة
    print("\n[4/4] حفظ النتائج...")
    pdf_name = Path(pdf_path).stem
    output_file = Path(output_dir) / f'{pdf_name}.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    
    print(f"✓ تم حفظ الملف: {output_file}")
    
    return output_file


def main():
    """
    الدالة الرئيسية
    """
    
    print("\n🚀 بدء معالجة PDF...\n")
    
    # مسار ملف PDF
    pdf_path = 'path/to/your/document.pdf'  # غير هذا إلى مسار ملفك
    
    # اختر طريقة المعالجة
    print("اختر طريقة المعالجة:")
    print("1. vLLM (سريع، يحتاج إعداد إضافي)")
    print("2. Transformers (بسيط، أبطأ)")
    
    choice = input("\nاختيارك (1 أو 2): ").strip()
    
    try:
        if choice == '1':
            print("\n⚠️  ملاحظة: هذا يتطلب إعداد vLLM والملفات المخصصة")
            output_file = process_pdf_with_vllm(pdf_path)
        else:
            output_file = process_pdf_simple(pdf_path)
        
        print(f"\n✅ تم بنجاح! الملف: {output_file}")
        
    except Exception as e:
        print(f"\n❌ خطأ: {str(e)}")
        print("\nتأكد من:")
        print("1. صحة مسار ملف PDF")
        print("2. تثبيت PyMuPDF: pip install PyMuPDF")
        print("3. توفر ذاكرة GPU كافية")


if __name__ == "__main__":
    main()


"""
مقارنة بين الطريقتين:
======================

1. vLLM:
   ✅ المميزات:
      - سرعة عالية جداً (~2500 رمز/ثانية)
      - معالجة متوازية للصفحات
      - استخدام فعال للذاكرة
   
   ❌ العيوب:
      - يحتاج إعداد إضافي
      - أكثر تعقيداً
      - يحتاج ملفات مخصصة

2. Transformers:
   ✅ المميزات:
      - سهل الاستخدام
      - لا يحتاج ملفات إضافية
      - مناسب للمبتدئين
   
   ❌ العيوب:
      - أبطأ بكثير
      - معالجة تسلسلية
      - استهلاك أعلى للذاكرة

نصائح للأداء الأمثل:
=====================

1. اختيار DPI المناسب:
   - 144 DPI: للمستندات العادية (موصى به)
   - 200 DPI: للمستندات ذات النصوص الصغيرة
   - 300 DPI: للمستندات عالية الجودة (بطيء)

2. إدارة الذاكرة:
   - للملفات الكبيرة (>50 صفحة): استخدم vLLM
   - قلل MAX_CONCURRENCY إذا نفدت الذاكرة
   - عالج الملف على دفعات إذا لزم الأمر

3. تحسين الجودة:
   - استخدم ملفات PDF أصلية (ليست ممسوحة ضوئياً)
   - تأكد من وضوح النص في PDF
   - استخدم crop_mode=True للصفحات الكبيرة

4. معالجة الأخطاء:
   - احفظ النتائج بشكل دوري
   - تعامل مع الصفحات الفاشلة بشكل منفصل
   - احتفظ بنسخة احتياطية من PDF الأصلي

مثال على الاستخدام المتقدم:
============================

# معالجة عدة ملفات PDF
pdf_files = [
    'document1.pdf',
    'document2.pdf',
    'document3.pdf'
]

for pdf_file in pdf_files:
    try:
        output = process_pdf_simple(pdf_file, f'./output/{Path(pdf_file).stem}')
        print(f"✓ {pdf_file} -> {output}")
    except Exception as e:
        print(f"✗ {pdf_file}: {str(e)}")
"""
