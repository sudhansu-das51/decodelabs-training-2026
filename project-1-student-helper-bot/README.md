# 🎓 Student Helper Bot

A rule-based AI chatbot built for college students to get quick help with exams, assignments, internships, career advice, and more.

> **Project 1 | DecodeLabs Industrial Training Kit | Batch 2026**

---

## 📌 What It Does

Student Helper Bot is a command-line chatbot that responds to keyword-based queries from college students. It uses a dictionary-driven knowledge base instead of complex AI models — making it fast, lightweight, and easy to extend.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or above
- No external libraries required (uses only Python built-ins)

### Run the Bot

```bash
python student_helper_bot.py
```

That's it. No installation, no setup.

---

## 💬 How to Use

Once the bot starts, just type a keyword or a sentence containing a keyword.

**Example inputs:**
```
You  : hello
You  : I have an exam tomorrow, help me
You  : tips for internship
You  : I'm feeling stressed
You  : help
You  : bye
```

To exit, type any of: `exit`, `quit`, `bye`, or `goodbye`.

---

## 🧠 Supported Topics

| Keyword       | What You Get                          |
|---------------|---------------------------------------|
| `hello` / `hi` / `hey` | Greeting response             |
| `exam`        | Exam preparation tips                 |
| `study`       | Smart study habits                    |
| `syllabus`    | Syllabus advice                       |
| `notes`       | Trusted resource links                |
| `assignment`  | Assignment strategy                   |
| `project`     | Project building tips                 |
| `deadline`    | Deadline management advice            |
| `internship`  | Where and how to find internships     |
| `resume`      | Fresher resume tips                   |
| `career`      | Year-wise career roadmap              |
| `placement`   | Placement preparation guide           |
| `attendance`  | Attendance advice                     |
| `cgpa`        | CGPA perspective                      |
| `stress`      | Mental wellness tips                  |
| `time`        | Shows current time                    |
| `date`        | Shows today's date                    |
| `resources`   | Top free learning websites            |
| `help`        | Lists all available topics            |

---

## 🗂️ Project Structure

```
student_helper_bot.py   ← Main file (entire project in one file)
README.md               ← This file
```

---

## ⚙️ How It Works (IPO Model)

```
Input → Sanitize → Match Keyword → Output Response
```

1. **Input** — User types a message
2. **Sanitize** — Converted to lowercase, whitespace stripped
3. **Match** — Exact match first, then keyword scan inside sentence
4. **Output** — Response printed; fallback if no match found

The knowledge base is a Python `dict`, giving O(1) lookup performance.

---

## 🧪 Test Results

All core functions tested and passing:

| Test | Status |
|------|--------|
| Syntax check | ✅ Pass |
| Imports (datetime, random) | ✅ Pass |
| `sanitize_input()` — uppercase, spaces, empty string | ✅ Pass |
| Exact keyword match | ✅ Pass |
| Keyword scan inside full sentence | ✅ Pass |
| Fallback for unknown input | ✅ Pass |
| Exit commands (exit, quit, bye, goodbye) | ✅ Pass |
| Ctrl+C graceful exit | ✅ Pass |

---

## 🔮 Possible Improvements (Future Scope)

- Add more topics (library, hostel, clubs, etc.)
- Connect to a GUI using `tkinter`
- Add fuzzy matching for typos
- Export chat history to a text file

---

## 👤 Author

Sudhansu Mohana Das 
DecodeLabs Industrial Training | Batch 2026

---

## 📄 License

This project is part of the DecodeLabs training curriculum. For educational use only.
