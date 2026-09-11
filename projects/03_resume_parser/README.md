# 03 — Resume Parser & Skill Matcher (Intermediate)

Regex + keyword extraction over plain-text resumes. Stdlib only.

```bash
cd projects/03_resume_parser
python main.py --resume sample_resume.txt
python main.py --resume sample_resume.txt --require python pytorch aws --min-years 3
```

Outputs JSON: emails, phones, matched skills, max years, and a
`SHORTLIST`/`REVIEW` verdict with score.

## Go production

Add PDFMiner text extraction + spaCy NER (`en_core_web_sm`) for names,
titles, and skill entities — keep the parse → match → verdict shape.
