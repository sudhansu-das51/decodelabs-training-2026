# Student Helper Bot — Project Documentation

**Project:** Student Helper Bot  
**Type:** Rule-Based Chatbot (CLI)  
**Batch:** DecodeLabs Industrial Training | 2026  
**Language:** Python 3.8+

---

## 1. Project Overview

Student Helper Bot is a keyword-driven chatbot that helps college students get instant advice on academics, career, and college life. It runs entirely in the terminal and requires no internet or external libraries.

---

## 2. Architecture

The project follows the **IPO (Input → Process → Output)** model taught in Module 01.

```
[ User Input ]
      ↓
[ sanitize_input() ]   ← lowercase + strip whitespace
      ↓
[ get_response() ]     ← dict lookup (O1) + keyword scan fallback
      ↓
[ Print Response ]
```

### Key Design Choice — Dictionary over if-elif

Instead of a long chain of `if-elif` statements, the bot uses a Python `dict` as its knowledge base. This gives O(1) lookup time and makes adding new topics as simple as adding one line to the dictionary.

---

## 3. File Breakdown

**`student_helper_bot.py`** — The only file in this project.

| Component | Purpose |
|-----------|---------|
| `KNOWLEDGE_BASE` | Dictionary of keyword → response pairs |
| `FALLBACK_RESPONSES` | List of random replies for unknown input |
| `sanitize_input()` | Cleans raw user input |
| `get_response()` | Finds the right response for a given input |
| `display_welcome()` | Prints the startup banner |
| `display_goodbye()` | Prints the exit message |
| `run_chatbot()` | Main loop — runs the bot |

---

## 4. Response Logic

`get_response()` works in two steps:

**Step 1 — Exact Match**  
Checks if the cleaned input is a direct key in the dictionary.  
Example: `"exam"` → returns exam tips immediately.

**Step 2 — Keyword Scan**  
If no exact match, scans each dictionary key to see if it appears anywhere inside the user's sentence.  
Example: `"how do i prepare for my exam tomorrow"` → finds `"exam"` inside the sentence → returns exam tips.

**Step 3 — Fallback**  
If nothing matches, returns a random message from `FALLBACK_RESPONSES` using `random.choice()`.

---

## 5. Error Handling

| Scenario | How It's Handled |
|----------|-----------------|
| Empty input | Prompts user to type something, loop continues |
| `Ctrl+C` / `EOF` | Caught with `try-except`, goodbye message shown, clean exit |
| Unknown topic | Random fallback response from `FALLBACK_RESPONSES` |

---

## 6. Testing Summary

Tested manually and with automated function-level tests.

**All 8 test cases passed:**

- Syntax validation — no errors
- Module imports — `datetime`, `random` both available
- `sanitize_input()` — handles uppercase, extra spaces, empty string correctly
- `get_response()` — exact match works
- `get_response()` — keyword inside sentence works
- `get_response()` — fallback triggers for unknown input
- Exit commands — `exit`, `quit`, `bye`, `goodbye` all recognized
- Graceful Ctrl+C handling — no crash

**No bugs or errors were found.** The code is clean and ready to submit.

---

## 7. How to Run

```bash
# No installation needed
python student_helper_bot.py
```

Tested on Python 3.8, 3.10, and 3.12.

---

## 8. Sample Interaction

```
============================================================
        🎓  STUDENT HELPER BOT  🎓
        Powered by DecodeLabs | Batch 2026
============================================================

  You  : hello
  Bot  : Hey there, student! 👋 I'm StudyBot. Ask me anything!

  You  : tips for my exam
  Bot  : 📚 Exam Tips:
           1. Start at least 1 week before the exam date.
           2. Use the Pomodoro technique — 25 min study, 5 min break.
           ...

  You  : bye
  Bot  : 👋 Thanks for using StudyBot! Keep learning, keep growing.
```

---

*Documentation prepared for DecodeLabs Project Submission | Batch 2026*
