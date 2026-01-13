import streamlit as st
import json, os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage
import pandas as pd
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try importing optional dependencies
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    try:
        import PyPDF2
        PYPDF2_AVAILABLE = True
    except ImportError:
        PYPDF2_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# -------------------- Setup --------------------
os.makedirs("reports", exist_ok=True)
os.makedirs("invoices", exist_ok=True)
os.makedirs("audits", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

st.set_page_config("AI Report & Invoice Generator", layout="wide")

# -------------------- Helpers --------------------
def create_pdf(text, file_id, folder, image_path=None):
    styles = getSampleStyleSheet()
    story = []

    if image_path:
        story.append(Image(image_path, width=300, height=200))
        story.append(Spacer(1, 20))

    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 10))

    path = f"{folder}/{file_id}.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    return open(path, "rb")

def list_files(folder, ext):
    return sorted([f for f in os.listdir(folder) if f.endswith(ext)], reverse=True)

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)

def generate_detailed_report(prompt, template):
    intro = {
        "Academic": "This academic report follows formal research standards.",
        "Business": "This business report supports strategic decision-making.",
        "Technical": "This technical report explains architecture and implementation.",
        "Summary": "This summary report gives a concise system overview."
    }

    sections = [
        "Title","Executive Summary","Introduction","Problem Statement",
        "Objectives of the System","System Architecture Description",
        "Technology Stack Used","Dataset Description and Handling",
        "Workflow and Methodology","Features and Functionalities",
        "Advantages","Limitations","Security and Data Privacy",
        "Future Enhancements","Conclusion"
    ]

    report = "### AI Generated Professional Report\n\n"
    report += f"**Template:** {template}\n\n"
    report += f"{intro[template]}\n\n"
    report += f"**Prompt:** {prompt}\n\n"

    for sec in sections:
        report += f"## {sec}\n"
        report += (
            "This section provides a detailed explanation with real-world relevance, "
            "structured workflow, and professional documentation standards. "
            "The explanation is suitable for academic and corporate evaluation.\n\n"
        )

    report += "### Final Conclusion\nThis system demonstrates automation and scalability."
    return report

def generate_invoice_json(prompt, template):
    return {
        "invoice_id": f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "date": datetime.now().strftime("%d-%m-%Y"),
        "invoice_template": template,
        "description": prompt,
        "amount": "₹8000",
        "tax": "18%",
        "total": "₹9440",
        "status": "Pending"
    }

# -------------------- File Extraction Functions --------------------
def extract_from_pdf(file_path):
    """Extract text and data from PDF file"""
    text = ""
    try:
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif PYPDF2_AVAILABLE:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        else:
            return {"text": "", "error": "No PDF library available"}
    except Exception as e:
        return {"text": "", "error": str(e)}
    
    return {"text": text, "error": None}

def extract_from_json(file_path):
    """Extract data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {"data": data, "error": None}
    except Exception as e:
        return {"data": {}, "error": str(e)}

def extract_from_csv(file_path):
    """Extract data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        data = {
            "columns": df.columns.tolist(),
            "rows": df.to_dict('records'),
            "row_count": len(df)
        }
        return {"data": data, "error": None}
    except Exception as e:
        return {"data": {}, "error": str(e)}

def extract_from_docx(file_path):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        return {"text": "", "error": "python-docx not available"}
    
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return {"text": text, "error": None}
    except Exception as e:
        return {"text": "", "error": str(e)}

def ai_extract_invoice_data(extracted_text, file_name):
    """AI-powered extraction of invoice data from PDF text"""
    
    # Prepare prompt for AI agent
    system_prompt = """You are an expert invoice data extraction agent. Extract structured data from invoice text and return ONLY valid JSON.
    
    Extract the following fields intelligently:
    - invoice_id: Invoice ID or number (look for patterns like INV-XXX, Invoice #, etc.)
    - date: Invoice date (any date format)
    - invoice_number: Invoice number if different from ID
    - vendor_name: Company/seller name issuing the invoice
    - vendor_address: Full address of vendor
    - client_name: Customer/buyer name
    - client_address: Full address of client
    - line_items: Array of items with description, quantity, unit_price, total
    - amounts: Object with subtotal, tax_rate, tax_amount, discount, total, currency
    - payment_info: Object with status, due_date, payment_terms, payment_method
    - metadata: Object with template_type, description, notes
    
    Return JSON in this EXACT structure:
    {
      "invoice_id": "string or null",
      "date": "string or null",
      "invoice_number": "string or null",
      "vendor_name": "string or null",
      "vendor_address": "string or null",
      "client_name": "string or null",
      "client_address": "string or null",
      "line_items": [{"description": "string", "quantity": number, "unit_price": number, "total": number}],
      "amounts": {"subtotal": number, "tax_rate": "string", "tax_amount": number, "discount": number, "total": number, "currency": "string"},
      "payment_info": {"status": "string", "due_date": "string or null", "payment_terms": "string or null", "payment_method": "string or null"},
      "metadata": {"template_type": "string or null", "description": "string", "notes": "string or null"}
    }
    
    If a field cannot be found, use null. Be intelligent about identifying vendor vs client, amounts, dates, etc."""
    
    # Limit text to avoid token limits
    limited_text = extracted_text[:4000] if len(extracted_text) > 4000 else extracted_text
    
    user_prompt = f"""Extract invoice data from this text:
    
    {limited_text}
    
    Return ONLY the JSON object, no explanations."""
    
    # Try using AI if available
    if OPENAI_AVAILABLE:
        try:
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1  # Low temperature for consistent extraction
                )
                ai_response = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                try:
                    # Remove markdown code blocks if present
                    if "```json" in ai_response:
                        ai_response = ai_response.split("```json")[1].split("```")[0].strip()
                    elif "```" in ai_response:
                        ai_response = ai_response.split("```")[1].split("```")[0].strip()
                    
                    extracted_data = json.loads(ai_response)
                    return extracted_data
                except json.JSONDecodeError:
                    st.warning(f"AI returned invalid JSON for {file_name}, using fallback extraction")
        except Exception as e:
            st.warning(f"AI extraction failed for {file_name}: {str(e)}, using fallback")
    
    # Fallback to regex-based extraction
    return parse_invoice_data_fallback(extracted_text, file_name)

def parse_invoice_data_fallback(extracted_text, file_name):
    """Fallback regex-based invoice parsing"""
    data = {
        "invoice_id": None,
        "date": None,
        "invoice_number": None,
        "vendor_name": None,
        "vendor_address": None,
        "client_name": None,
        "client_address": None,
        "line_items": [],
        "amounts": {
            "subtotal": None,
            "tax_rate": None,
            "tax_amount": None,
            "discount": 0,
            "total": None,
            "currency": "INR"
        },
        "payment_info": {
            "status": "Pending",
            "due_date": None,
            "payment_terms": None,
            "payment_method": None
        },
        "metadata": {
            "template_type": None,
            "description": extracted_text[:500] if extracted_text else "",
            "notes": None
        }
    }
    
    # Extract invoice ID
    inv_id_match = re.search(r'(?:invoice|inv)[\s#:-]*([A-Z0-9-]+)', extracted_text, re.IGNORECASE)
    if inv_id_match:
        data["invoice_id"] = inv_id_match.group(1).strip()
    
    # Extract date
    date_patterns = [
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'(?:date|dated)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, extracted_text, re.IGNORECASE)
        if date_match:
            data["date"] = date_match.group(1)
            break
    
    # Extract total amount
    total_patterns = [
        r'total[:\s]*[₹$€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'grand\s*total[:\s]*[₹$€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
    ]
    for pattern in total_patterns:
        amount_match = re.search(pattern, extracted_text, re.IGNORECASE)
        if amount_match:
            try:
                amount_str = amount_match.group(1).replace(',', '')
                data["amounts"]["total"] = float(amount_str)
            except:
                pass
            break
    
    return data

def ai_extract_audit_data(extracted_text, file_name):
    """AI-powered extraction of audit data from PDF text"""
    
    # Prepare prompt for AI agent
    system_prompt = """You are an expert audit report data extraction agent. Extract structured data from audit text and return ONLY valid JSON.
    
    Extract the following fields intelligently:
    - audit_id: Audit ID or reference number
    - audit_date: Date of audit
    - audit_period: Period covered (e.g., Q4 2025, Jan-Mar 2026)
    - auditor_name: Name of auditing firm/person
    - auditee_name: Name of organization being audited
    - findings: Array of findings with finding_id, category, severity, description, recommendation
    - compliance_status: Object with overall_score, compliant_items, non_compliant_items, compliance_percentage
    - risk_assessment: Object with overall_risk_level, high_risks, medium_risks, low_risks
    - metadata: Object with audit_type, scope, standards
    
    Return JSON in this EXACT structure:
    {
      "audit_id": "string or null",
      "audit_date": "string or null",
      "audit_period": "string or null",
      "auditor_name": "string or null",
      "auditee_name": "string or null",
      "findings": [{"finding_id": "string", "category": "string", "severity": "string", "description": "string", "recommendation": "string"}],
      "compliance_status": {"overall_score": number, "compliant_items": number, "non_compliant_items": number, "compliance_percentage": number},
      "risk_assessment": {"overall_risk_level": "string", "high_risks": number, "medium_risks": number, "low_risks": number},
      "metadata": {"audit_type": "string or null", "scope": "string or null", "standards": ["string"]}
    }
    
    If a field cannot be found, use null. Be intelligent about identifying findings, compliance scores, risk levels, etc."""
    
    # Limit text to avoid token limits
    limited_text = extracted_text[:4000] if len(extracted_text) > 4000 else extracted_text
    
    user_prompt = f"""Extract audit data from this text:
    
    {limited_text}
    
    Return ONLY the JSON object, no explanations."""
    
    # Try using AI if available
    if OPENAI_AVAILABLE:
        try:
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1  # Low temperature for consistent extraction
                )
                ai_response = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                try:
                    # Remove markdown code blocks if present
                    if "```json" in ai_response:
                        ai_response = ai_response.split("```json")[1].split("```")[0].strip()
                    elif "```" in ai_response:
                        ai_response = ai_response.split("```")[1].split("```")[0].strip()
                    
                    extracted_data = json.loads(ai_response)
                    return extracted_data
                except json.JSONDecodeError:
                    st.warning(f"AI returned invalid JSON for {file_name}, using fallback extraction")
        except Exception as e:
            st.warning(f"AI extraction failed for {file_name}: {str(e)}, using fallback")
    
    # Fallback to regex-based extraction
    return parse_audit_data_fallback(extracted_text, file_name)

def parse_audit_data_fallback(extracted_text, file_name):
    """Fallback regex-based audit parsing"""
    data = {
        "audit_id": None,
        "audit_date": None,
        "audit_period": None,
        "auditor_name": None,
        "auditee_name": None,
        "findings": [],
        "compliance_status": {
            "overall_score": None,
            "compliant_items": 0,
            "non_compliant_items": 0,
            "compliance_percentage": None
        },
        "risk_assessment": {
            "overall_risk_level": "Medium",
            "high_risks": 0,
            "medium_risks": 0,
            "low_risks": 0
        },
        "metadata": {
            "audit_type": None,
            "scope": None,
            "standards": []
        }
    }
    
    # Extract audit ID
    audit_id_match = re.search(r'(?:audit|aud)[\s#:-]*([A-Z0-9-]+)', extracted_text, re.IGNORECASE)
    if audit_id_match:
        data["audit_id"] = audit_id_match.group(1).strip()
    
    return data

def build_json_structure(file, file_type, folder):
    """Build standardized JSON structure from uploaded file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_id = f"{file_type}_{timestamp}"
    file_name = file.name
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Save original file
    file_path = f"{folder}/{file_id}_{file_name}"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Extract data based on file type
    extracted_data = {}
    extraction_method = None
    
    if file_ext == '.pdf':
        extraction_method = "ai_pdf_parser"
        result = extract_from_pdf(file_path)
        if result.get("error"):
            extracted_data = {"error": result["error"], "text": result.get("text", "")}
        else:
            # Use AI agent to extract structured data
            if file_type == "invoice":
                extracted_data = ai_extract_invoice_data(result["text"], file_name)
            else:
                extracted_data = ai_extract_audit_data(result["text"], file_name)
    
    elif file_ext == '.json':
        extraction_method = "json_parser"
        result = extract_from_json(file_path)
        if result.get("error"):
            extracted_data = {"error": result["error"]}
        else:
            # Handle both dict and list JSON structures
            json_data = result["data"]
            if isinstance(json_data, list):
                # If it's a list, wrap it in a dict
                extracted_data = {"items": json_data, "count": len(json_data)}
            else:
                extracted_data = json_data
    
    elif file_ext == '.csv':
        extraction_method = "csv_parser"
        result = extract_from_csv(file_path)
        if result.get("error"):
            extracted_data = {"error": result["error"]}
        else:
            extracted_data = result["data"]
    
    elif file_ext in ['.docx', '.doc']:
        extraction_method = "ai_docx_parser"
        result = extract_from_docx(file_path)
        if result.get("error"):
            extracted_data = {"error": result["error"], "text": result.get("text", "")}
        else:
            # Use AI agent to extract structured data
            if file_type == "invoice":
                extracted_data = ai_extract_invoice_data(result["text"], file_name)
            else:
                extracted_data = ai_extract_audit_data(result["text"], file_name)
    
    # Build standardized JSON structure matching the design specification
    # Calculate extracted and missing fields
    extracted_fields = []
    missing_fields = []
    
    if isinstance(extracted_data, dict) and not extracted_data.get("error"):
        # For invoices, check expected fields
        if file_type == "invoice":
            expected_fields = ["invoice_id", "date", "invoice_number", "vendor_name", "vendor_address", 
                             "client_name", "client_address", "line_items", "amounts", "payment_info", "metadata"]
            for field in expected_fields:
                if field in extracted_data and extracted_data[field] is not None:
                    extracted_fields.append(field)
                else:
                    missing_fields.append(field)
        # For audits, check expected fields
        elif file_type == "audit":
            expected_fields = ["audit_id", "audit_date", "audit_period", "auditor_name", "auditee_name",
                             "findings", "compliance_status", "risk_assessment", "metadata"]
            for field in expected_fields:
                if field in extracted_data and extracted_data[field] is not None:
                    extracted_fields.append(field)
                else:
                    missing_fields.append(field)
        else:
            extracted_fields = list(extracted_data.keys())
    elif isinstance(extracted_data, list):
        # Handle list structures
        extracted_fields = ["items", "count"]
        confidence_score = 0.8
    
    # Calculate confidence score based on extraction quality
    if not hasattr(locals(), 'confidence_score'):
        if extracted_data.get("error") if isinstance(extracted_data, dict) else False:
            confidence_score = 0.0
        elif isinstance(extracted_data, dict):
            # Higher confidence if more fields extracted
            total_expected = len(extracted_fields) + len(missing_fields)
            if total_expected > 0:
                confidence_score = len(extracted_fields) / total_expected
            else:
                confidence_score = 0.85  # Default if structure is different
        else:
            confidence_score = 0.0
    
    json_structure = {
        "file_id": file_id,
        "file_name": file_name,
        "file_type": file_type,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": file_path,
        "extracted_data": extracted_data,
        "extraction_metadata": {
            "extraction_method": extraction_method,
            "confidence_score": round(confidence_score, 2),
            "extracted_fields": extracted_fields,
            "missing_fields": missing_fields,
            "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save JSON structure
    json_path = f"{folder}/{file_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, indent=4, ensure_ascii=False)
    
    return json_structure

# -------------------- AI Agent Integration --------------------
def call_ai_agent(user_prompt, uploaded_files_data):
    """Call AI agent to generate report based on prompt and data"""
    
    # Prepare context for AI agent
    context = {
        "user_prompt": user_prompt,
        "uploaded_files": uploaded_files_data,
        "total_files": len(uploaded_files_data),
        "file_types": [f["file_type"] for f in uploaded_files_data]
    }
    
    # Build prompt for AI
    system_prompt = """You are an expert financial and audit report analyst. 
    Generate a comprehensive, professional report based on the user's requirements and the provided data.
    Follow the structure from consolidated report guidelines:
    - Executive Summary
    - Financial Summary (if applicable)
    - Status Analysis
    - Individual Details
    - Key Metrics & Statistics
    - Recommendations
    - Conclusion
    
    Make the report detailed, accurate, and suitable for professional use."""
    
    user_message = f"""User Request: {user_prompt}

Available Data:
{json.dumps(context, indent=2)}

Please generate a comprehensive report based on the user's requirements and the provided invoice/audit data.
Include all relevant sections, analysis, and recommendations."""
    
    # If OpenAI is available, use it
    if OPENAI_AVAILABLE:
        try:
            # Note: User needs to set OPENAI_API_KEY in environment or Streamlit secrets
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                st.info("🤖 Using OpenAI API to generate report...")
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                st.success("✅ Report generated using AI agent (OpenAI)")
                return response.choices[0].message.content
            else:
                st.warning("⚠️ OpenAI API key not found. Using template-based generation.")
        except Exception as e:
            st.warning(f"❌ AI API call failed: {str(e)}. Using template-based generation.")
    else:
        st.info("ℹ️ OpenAI library not available. Using template-based generation.")
    
    # Fallback: Generate report using template and data analysis
    st.info("📝 Generating report using template-based fallback method...")
    return generate_ai_report_fallback(user_prompt, uploaded_files_data)

def generate_ai_report_fallback(user_prompt, uploaded_files_data):
    """Fallback report generation when AI agent is not available"""
    report = f"# AI-Generated Consolidated Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n"
    report += f"**User Request:** {user_prompt}\n\n"
    
    report += "## Executive Summary\n\n"
    report += f"This report analyzes {len(uploaded_files_data)} uploaded file(s). "
    report += "The analysis is based on extracted data from invoices and audit documents.\n\n"
    
    # Financial Summary
    total_amount = 0
    invoice_count = 0
    audit_count = 0
    
    for file_data in uploaded_files_data:
        if file_data["file_type"] == "invoice":
            invoice_count += 1
            extracted = file_data.get("extracted_data", {})
            if isinstance(extracted, dict):
                total = extracted.get("total")
                if total:
                    try:
                        total_amount += float(str(total).replace('₹', '').replace(',', ''))
                    except:
                        pass
        else:
            audit_count += 1
    
    report += "## Financial Summary\n\n"
    if invoice_count > 0:
        report += f"- **Total Invoices Analyzed:** {invoice_count}\n"
        report += f"- **Total Amount:** ₹{total_amount:,.2f}\n"
        if invoice_count > 0:
            report += f"- **Average Invoice Amount:** ₹{total_amount/invoice_count:,.2f}\n\n"
    
    if audit_count > 0:
        report += f"- **Total Audits Analyzed:** {audit_count}\n\n"
    
    # Individual Details
    report += "## Individual File Details\n\n"
    for idx, file_data in enumerate(uploaded_files_data, 1):
        report += f"### File {idx}: {file_data['file_name']}\n\n"
        report += f"- **File ID:** {file_data['file_id']}\n"
        report += f"- **File Type:** {file_data['file_type']}\n"
        report += f"- **Upload Date:** {file_data['upload_date']}\n\n"
        
        extracted = file_data.get("extracted_data", {})
        if isinstance(extracted, dict):
            for key, value in extracted.items():
                if key != "error" and value:
                    report += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        report += "\n"
    
    report += "## Recommendations\n\n"
    report += "1. Review all pending invoices for payment follow-up\n"
    report += "2. Address any compliance issues identified in audit reports\n"
    report += "3. Implement regular reporting schedule for better tracking\n\n"
    
    report += "## Conclusion\n\n"
    report += "This consolidated report provides a comprehensive overview of the uploaded documents. "
    report += "Regular monitoring and analysis of financial and audit data is recommended for effective business management.\n"
    
    return report

# -------------------- UI --------------------
st.title("🚀 AI Report & Invoice Generator")

left, right = st.columns([2,1])

with left:
    tab1, tab2, tab3 = st.tabs(["🧾 Report Generator", "💰 Invoice Generator", "🤖 AI Consolidated Report"])

    # -------- REPORT --------
    with tab1:
        st.header("Generate Professional Report")

        template = st.selectbox(
            "Select Report Template",
            ["Academic", "Business", "Technical", "Summary"]
        )

        prompt = st.text_area("Enter Report Prompt")
        uploaded_file = st.file_uploader(
            "Upload PDF / JSON / CSV / DOCX (Optional)",
            type=["pdf","json","csv","docx"]
        )

        image_file = st.file_uploader(
            "Upload Image (Optional)",
            type=["png","jpg","jpeg"]
        )

        if st.button("Generate Report"):
            if not prompt:
                st.error("Please enter report prompt")
            else:
                image_path = None
                if image_file:
                    image_path = f"uploads/{image_file.name}"
                    PILImage.open(image_file).save(image_path)

                report = generate_detailed_report(prompt, template)
                report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                open(f"reports/{report_id}.txt","w",encoding="utf-8").write(report)

                st.session_state["last_report"] = report_id
                st.session_state["report_image"] = image_path
                st.success("✅ Report Generated")

        if "last_report" in st.session_state:
            file = st.session_state["last_report"]
            text = open(f"reports/{file}.txt").read()
            st.markdown(text)

            pdf = create_pdf(text, file, "reports", st.session_state.get("report_image"))
            st.download_button("⬇ Download PDF", pdf, f"{file}.pdf")

    # -------- INVOICE --------
    with tab2:
        st.header("Professional Invoice Generator")

        template = st.selectbox(
            "Select Invoice Template",
            ["Academic", "Business", "Technical", "Summary"],
            key="inv_temp"
        )

        invoice_prompt = st.text_area("Enter Invoice Requirements")
        uploaded_invoice = st.file_uploader(
            "Upload PDF / JSON (Optional)",
            type=["pdf","json"]
        )

        image_file = st.file_uploader(
            "Upload Invoice Image (Optional)",
            type=["png","jpg","jpeg"],
            key="inv_img"
        )

        if st.button("Generate Invoice"):
            if not invoice_prompt:
                st.error("Enter invoice details")
            else:
                image_path = None
                if image_file:
                    image_path = f"uploads/{image_file.name}"
                    PILImage.open(image_file).save(image_path)

                data = generate_invoice_json(invoice_prompt, template)
                invoice_id = f"invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                json.dump(data, open(f"invoices/{invoice_id}.json","w"), indent=4)

                st.session_state["last_invoice"] = invoice_id
                st.session_state["invoice_image"] = image_path
                st.success("✅ Invoice Generated")

        if "last_invoice" in st.session_state:
            file = st.session_state["last_invoice"]
            text = open(f"invoices/{file}.json").read()
            st.code(text, language="json")

            pdf = create_pdf(text, file, "invoices", st.session_state.get("invoice_image"))
            st.download_button("⬇ Download PDF", pdf, f"{file}.pdf")

    # -------- AI CONSOLIDATED REPORT --------
    with tab3:
        st.header("🤖 AI-Powered Consolidated Report Generator")
        st.markdown("Upload invoices/audits and let AI generate a comprehensive report!")
        
        # File type selection
        file_type = st.radio(
            "Select file type:",
            ["Invoice", "Audit", "Both"],
            horizontal=True
        )
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Invoice/Audit Files (PDF, JSON, CSV, DOCX)",
            type=["pdf", "json", "csv", "docx"],
            accept_multiple_files=True,
            key="consolidated_files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Display uploaded files
            with st.expander("📁 View Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size} bytes)")
        
        # Report prompt
        report_prompt = st.text_area(
            "Enter Report Requirements",
            placeholder="Example: Generate a financial summary report showing total revenue, payment status breakdown, and monthly trends for all uploaded invoices.",
            height=100,
            key="consolidated_prompt"
        )
        
        # Generate button
        if st.button("🚀 Generate AI Report", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one file")
            elif not report_prompt:
                st.error("Please enter report requirements")
            else:
                with st.spinner("Processing files and generating report..."):
                    # Process uploaded files
                    processed_files = []
                    folder_map = {
                        "Invoice": "invoices",
                        "Audit": "audits",
                        "Both": "invoices"  # Default, will handle both
                    }
                    
                    for file in uploaded_files:
                        # Determine file type for "Both" option
                        if file_type == "Both":
                            # Simple heuristic: check filename
                            file_lower = file.name.lower()
                            if "audit" in file_lower:
                                current_folder = "audits"
                                current_type = "audit"
                            else:
                                current_folder = "invoices"
                                current_type = "invoice"
                        else:
                            current_folder = folder_map[file_type]
                            current_type = file_type.lower()
                        
                        try:
                            json_data = build_json_structure(file, current_type, current_folder)
                            processed_files.append(json_data)
                            st.success(f"✅ Processed: {file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                    
                    if processed_files:
                        # Call AI agent
                        st.info("🤖 Generating report with AI agent...")
                        report_content = call_ai_agent(report_prompt, processed_files)
                        
                        # Save report
                        report_id = f"consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        report_txt_path = f"reports/{report_id}.txt"
                        with open(report_txt_path, "w", encoding="utf-8") as f:
                            f.write(report_content)
                        
                        # Store in session
                        st.session_state["last_consolidated_report"] = report_id
                        st.session_state["consolidated_files_data"] = processed_files
                        st.success("✅ AI Report Generated Successfully!")
        
        # Display generated report
        if "last_consolidated_report" in st.session_state:
            report_id = st.session_state["last_consolidated_report"]
            report_path = f"reports/{report_id}.txt"
            
            if os.path.exists(report_path):
                st.markdown("---")
                st.subheader("📄 Generated Report")
                
                with open(report_path, "r", encoding="utf-8") as f:
                    report_text = f.read()
                
                st.markdown(report_text)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    pdf_file = create_pdf(report_text, report_id, "reports")
                    st.download_button(
                        "⬇ Download PDF",
                        pdf_file,
                        f"{report_id}.pdf",
                        key="download_consolidated_pdf"
                    )
                
                with col2:
                    st.download_button(
                        "⬇ Download TXT",
                        report_text,
                        f"{report_id}.txt",
                        mime="text/plain",
                        key="download_consolidated_txt"
                    )

# -------------------- HISTORY --------------------
with right:
    st.subheader("📚 History")

    st.markdown("### Reports")
    for f in list_files("reports",".txt"):
        col1, col2 = st.columns([3,1])
        if col1.button(f):
            st.session_state["last_report"] = f.replace(".txt","")
        if col2.button("❌", key=f"del_{f}"):
            delete_file(f"reports/{f}")

    st.markdown("### Invoices")
    for f in list_files("invoices",".json"):
        col1, col2 = st.columns([3,1])
        if col1.button(f, key=f"in_{f}"):
            st.session_state["last_invoice"] = f.replace(".json","")
        if col2.button("❌", key=f"del_in_{f}"):
            delete_file(f"invoices/{f}")
