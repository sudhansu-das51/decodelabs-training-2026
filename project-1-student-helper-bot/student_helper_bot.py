# =============================================================================
#  STUDENT HELPER BOT — Rule-Based AI Chatbot
#  Project 1 | DecodeLabs Industrial Training Kit | Batch 2026
#  Developer  : [Your Name]
#  Description: A deterministic, rule-based chatbot built for college students.
#               Uses a dictionary-driven knowledge base (O(1) lookup) instead
#               of if-elif ladders, following the IPO model taught in class.
# =============================================================================


import datetime
import random


# -----------------------------------------------------------------------------
#  KNOWLEDGE BASE — The Brain of the Bot
#  Structure : { trigger_keyword : response_string }
#  Why dict?  : O(1) lookup vs O(n) if-elif ladder (as taught in Module 01)
# -----------------------------------------------------------------------------

KNOWLEDGE_BASE: dict[str, str] = {

    # --- Greetings ---
    "hello"       : "Hey there, student! 👋 I'm StudyBot. Ask me anything about college life!",
    "hi"          : "Hi! Ready to help you ace your college journey. What's on your mind?",
    "hey"         : "Hey! What can I help you with today?",
    "good morning": "Good morning! ☀️ Hope you're ready for a productive day. How can I help?",
    "good night"  : "Good night! 🌙 Don't forget to revise before you sleep. Rest well!",

    # --- Exams & Study ---
    "exam"        : (
        "📚 Exam Tips:\n"
        "  1. Start at least 1 week before the exam date.\n"
        "  2. Use the Pomodoro technique — 25 min study, 5 min break.\n"
        "  3. Solve previous year question papers.\n"
        "  4. Teach what you learned to someone else — best revision trick!\n"
        "  Good luck! You've got this. 💪"
    ),
    "study"       : (
        "📖 Smart Study Habits:\n"
        "  → Active recall > passive reading.\n"
        "  → Use mind maps for complex topics.\n"
        "  → Study in 90-minute deep work blocks.\n"
        "  → Switch subjects every session to stay fresh."
    ),
    "syllabus"    : (
        "📋 Always check your official university portal for the latest syllabus.\n"
        "  Tip: Download it on Day 1, highlight the high-weightage topics first!"
    ),
    "notes"       : (
        "📝 For notes, check out these trusted resources:\n"
        "  → GeeksforGeeks  : geeksforgeeks.org\n"
        "  → NPTEL Lectures : nptel.ac.in\n"
        "  → YouTube Channels: Apna College, CodeWithHarry, Jenny's Lectures"
    ),

    # --- Assignments & Projects ---
    "assignment"  : (
        "📌 Assignment Strategy:\n"
        "  1. Read the problem statement at least twice.\n"
        "  2. Break it into smaller subtasks.\n"
        "  3. Never copy-paste — paraphrase and understand.\n"
        "  4. Submit at least a day before the deadline!"
    ),
    "project"     : (
        "🛠️ Project Tips:\n"
        "  → Pick a problem you genuinely face in college life.\n"
        "  → Keep scope small but execution clean.\n"
        "  → Always write a README for your project.\n"
        "  → Use GitHub to showcase your work!"
    ),
    "deadline"    : "⏰ Deadline approaching? Break your work into 3 parts: Plan → Execute → Review. Don't panic, start now!",

    # --- Career & Internships ---
    "internship"  : (
        "💼 Internship Hunting:\n"
        "  → LinkedIn, Internshala, Unstop — check daily.\n"
        "  → Apply to 10 per week minimum.\n"
        "  → Tailor your resume for each role.\n"
        "  → A personal project > a blank resume every single time."
    ),
    "resume"      : (
        "📄 Resume Tips:\n"
        "  → Keep it to 1 page as a fresher.\n"
        "  → Lead with a strong Skills section.\n"
        "  → Add GitHub and LinkedIn links.\n"
        "  → Use action verbs: Built, Developed, Implemented, Optimized."
    ),
    "career"      : (
        "🚀 Career Roadmap for CS Students:\n"
        "  Year 1-2 : Learn DSA + one programming language deeply.\n"
        "  Year 2-3 : Build projects, contribute to open source.\n"
        "  Year 3-4 : Internships + competitive coding + placements prep."
    ),
    "placement"   : (
        "🎯 Placement Preparation:\n"
        "  → DSA on LeetCode (150 problems is enough to start).\n"
        "  → Practice aptitude on IndiaBix.\n"
        "  → Mock interviews with peers.\n"
        "  → Research the company before every interview!"
    ),

    # --- College Life ---
    "attendance"  : "📊 Attendance below 75%? Talk to your professor early. Most colleges allow medical/duty leave if communicated properly.",
    "cgpa"        : (
        "📈 CGPA Advice:\n"
        "  → Don't obsess, but don't ignore it either.\n"
        "  → A 7.5+ CGPA keeps most doors open.\n"
        "  → Projects and skills matter more at top product companies."
    ),
    "stress"      : (
        "🧘 Feeling stressed? Totally normal in college.\n"
        "  → Take a 10-minute walk — seriously, it works.\n"
        "  → Talk to a friend or mentor.\n"
        "  → Sleep 7-8 hours. Your brain consolidates memory during sleep.\n"
        "  → Remember: one bad exam does NOT define your career. 💙"
    ),
    "time"        : f"🕒 Current time is: {datetime.datetime.now().strftime('%I:%M %p')}",
    "date"        : f"📅 Today's date is: {datetime.datetime.now().strftime('%A, %d %B %Y')}",

    # --- Resources ---
    "resources"   : (
        "🌐 Top Free Learning Resources:\n"
        "  → CS Fundamentals : cs50.harvard.edu (free!)\n"
        "  → Coding Practice  : leetcode.com, codeforces.com\n"
        "  → Web Dev          : freecodecamp.org\n"
        "  → AI / ML          : fast.ai, kaggle.com\n"
        "  → Open Courses     : nptel.ac.in, coursera.org"
    ),

    # --- Help & Exit ---
    "help"        : (
        "🤖 I can help you with topics like:\n"
        "  exam, study, syllabus, notes, assignment, project,\n"
        "  deadline, internship, resume, career, placement,\n"
        "  attendance, cgpa, stress, time, date, resources.\n\n"
        "  Just type any of these keywords!"
    ),
}


# -----------------------------------------------------------------------------
#  FALLBACK RESPONSES — Returned when intent is not recognized
#  Using random.choice() gives the bot a more natural feel
# -----------------------------------------------------------------------------

FALLBACK_RESPONSES: list[str] = [
    "🤔 Hmm, I didn't quite catch that. Type 'help' to see what I can do!",
    "❓ I'm not sure about that one. Try rephrasing or type 'help'.",
    "📭 That's outside my knowledge base right now. Type 'help' for available topics!",
    "🙈 Oops! I don't have an answer for that yet. Try 'help' to explore what I know.",
]


# -----------------------------------------------------------------------------
#  CORE FUNCTIONS
# -----------------------------------------------------------------------------

def sanitize_input(raw: str) -> str:
    """
    Phase 1 — Input Sanitization (as per IPO Model).
    Converts raw user input to lowercase and strips leading/trailing whitespace.
    This ensures 'Hello', 'HELLO', '  hello  ' all map to the same key: 'hello'.
    """
    return raw.lower().strip()


def get_response(clean_input: str) -> str:
    """
    Phase 2 — Intent Matching & Response Generation (as per IPO Model).
    Uses dictionary .get() for O(1) lookup with a random fallback.
    No if-elif ladder — scalable, clean, and maintainable.
    """

    # --- Direct exact-match lookup ---
    if clean_input in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[clean_input]

    # --- Keyword scan — checks if any known key appears inside user's sentence ---
    # Handles inputs like "how do i prepare for my exam?" → matches "exam"
    for keyword in KNOWLEDGE_BASE:
        if keyword in clean_input:
            return KNOWLEDGE_BASE[keyword]

    # --- Fallback — unknown intent ---
    return random.choice(FALLBACK_RESPONSES)


def display_welcome() -> None:
    """Prints the welcome banner when the bot starts."""
    print("\n" + "=" * 60)
    print("        🎓  STUDENT HELPER BOT  🎓")
    print("        Powered by DecodeLabs | Batch 2026")
    print("=" * 60)
    print("  Hi! I'm StudyBot — your personal college assistant.")
    print("  Type 'help' to see what I can do.")
    print("  Type 'exit' or 'quit' anytime to leave.")
    print("=" * 60 + "\n")


def display_goodbye() -> None:
    """Prints the goodbye message when the user exits."""
    print("\n" + "-" * 60)
    print("  👋  Thanks for using StudyBot! Keep learning, keep growing.")
    print("  📌  DecodeLabs — Building the next generation of engineers.")
    print("-" * 60 + "\n")


# -----------------------------------------------------------------------------
#  MAIN ENGINE — The Continuous Input Loop (Heartbeat of the Bot)
# -----------------------------------------------------------------------------

def run_chatbot() -> None:
    """
    The main control loop — runs indefinitely until an exit command is received.
    This is the 'while True' infinite loop pattern described in Module 01.
    """
    display_welcome()

    EXIT_COMMANDS: set[str] = {"exit", "quit", "bye", "goodbye"}

    while True:

        # --- Phase 1: Input & Sanitization ---
        try:
            raw_input_text = input("  You  : ")
        except (KeyboardInterrupt, EOFError):
            # Handles Ctrl+C gracefully — no ugly crash
            display_goodbye()
            break

        clean_input = sanitize_input(raw_input_text)

        # --- Guard: ignore empty input ---
        if not clean_input:
            print("  Bot  : Please type something! (or 'help' to see topics)\n")
            continue

        # --- Exit Strategy: Clean Kill Command ---
        if clean_input in EXIT_COMMANDS:
            display_goodbye()
            break

        # --- Phase 2 & 3: Process → Output ---
        response = get_response(clean_input)
        print(f"\n  Bot  : {response}\n")


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_chatbot()
