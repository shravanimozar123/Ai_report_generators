import streamlit as st
import json, os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage

# -------------------- Setup --------------------
os.makedirs("reports", exist_ok=True)
os.makedirs("invoices", exist_ok=True)
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

# -------------------- UI --------------------
st.title("🚀 AI Report & Invoice Generator")

left, right = st.columns([2,1])

with left:
    tab1, tab2 = st.tabs(["🧾 Report Generator", "💰 Invoice Generator"])

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
