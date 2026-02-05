# Instructions for Creating PDF Report

To create the project report PDF from PROJECT_REPORT.md:

## Method 1: Using Online Markdown to PDF Converter

1. Open PROJECT_REPORT.md in any text editor
2. Visit https://www.markdowntopdf.com/ or https://md2pdf.netlify.app/
3. Copy and paste the content from PROJECT_REPORT.md
4. Click "Convert to PDF"
5. Download the generated PDF

## Method 2: Using VS Code

1. Install "Markdown PDF" extension in VS Code
2. Open PROJECT_REPORT.md in VS Code
3. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
4. Type "Markdown PDF: Export (pdf)" and press Enter
5. PDF will be saved in the same directory

## Method 3: Using Pandoc (Command Line)

```bash
# Install pandoc
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended texlive-latex-extra

# Convert to PDF
pandoc PROJECT_REPORT.md -o PROJECT_REPORT.pdf --pdf-engine=pdflatex
```

## Method 4: Using Python (pypandoc)

```bash
pip install pypandoc

python3 << EOF
import pypandoc
output = pypandoc.convert_file('PROJECT_REPORT.md', 'pdf', outputfile="PROJECT_REPORT.pdf")
EOF
```

## Adding Screenshots to PDF

The PROJECT_REPORT.md file has placeholders for screenshots. Before converting to PDF:

1. Take screenshots of:
   - Initial page with model metrics
   - Form filled with sample data
   - Prediction results displayed

2. Save screenshots in /app directory

3. Update PROJECT_REPORT.md with image references:
   ```markdown
   ![Screenshot 1](./screenshot1.png)
   ```

4. Then convert to PDF using any method above

## What to Include in Final PDF

The PROJECT_REPORT.md already includes:

1. Project Overview
2. System Architecture (with ASCII diagram)
3. Dataset Description (Pima Indians Diabetes Dataset)
4. Data Preprocessing Steps
5. Model Implementation (Logistic Regression + Random Forest)
6. Model Comparison Table
7. Evaluation Metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
8. User Interface Design
9. Results Section (with screenshot placeholders)
10. Conclusion
11. Steps to Run the Project
12. Appendix with additional technical details

All sections are complete and ready for conversion to PDF format.
