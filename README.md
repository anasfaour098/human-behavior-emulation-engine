🧠 Human-Behavior Emulation Engine: A Deep Dive into Non-Deterministic Automation
🚀 1. Introduction: The Philosophy of Emulation
Traditional web automation has long relied on static scripts—sequential, predictable, and easily detectable by modern heuristic analysis. As platforms like Facebook implement advanced AI-driven bot detection, the challenge shifts from "how to automate" to "how to emulate."
This project, the HumanizedFB Engine, represents a paradigm shift. It is not a mere automation tool; it is a research-oriented framework designed to study and replicate the nuances of human interaction within restricted digital environments. By integrating mathematical models, psychological behavioral states, and advanced browser fingerprinting bypasses, this engine explores the boundary between machine precision and human unpredictability.
🛠️ 2. The Core Architecture: State-Driven Logic
At the heart of the engine lies a Finite State Machine (FSM). Unlike standard loops, the FSM allows the bot to transition between logical states based on environmental triggers and internal "moods".
🎭 A. Behavioral States
The engine operates across several distinct states defined in the State class:
 * IDLING: Simulates a user who has left the tab open but is not actively engaging. Uses a mean sleep time of 7.0 seconds to mimic passive presence.
 * BROWSING: Active content consumption, characterized by scrolling and occasional hovers.
 * SEARCHING: Goal-oriented behavior where the engine targets specific keywords provided by the user.
 * DISTRACTED: A unique state where the bot deviates from its primary task to mimic human curiosity or short attention spans.
 * EXITING_ABRUPTLY: Simulates a user suddenly closing the browser, preventing predictable session termination patterns.
🌡️ B. The Mood System
To ensure no two sessions are identical, the engine selects a SessionMood upon launch, with specific probability weights:
 * LAZY (31%): Characterized by long idle times and minimal clicks.
 * CURIOUS (31%): Increases the probability of searching and exploring new group links.
 * SOCIAL (25%): Prioritizes interactions like likes and simulated viewing of comments.
 * PRODUCTIVE (13%): Focuses on the user’s primary goals, such as group joins and posting.
📐 3. Mathematical Emulation of Human Flaws
Humans are inherently "noisy" and inefficient. This project digitizes that inefficiency through two main mathematical concepts:
🖱️ 1. Bézier Curve Mouse Movements
Standard Selenium commands move the mouse in a straight line—a major red flag. Our _bezier_mouse_move function utilizes cubic Bézier curves to generate paths with natural arcs, jitters, and varying acceleration.
 * Control Points: The algorithm generates random control points to ensure every movement path is unique.
 * Micro-Jitters: Small, high-frequency oscillations are added to simulate the physiological tremors of a human hand.
⏳ 2. Gaussian Distribution for Temporal Logic
Instead of using fixed delays (e.g., time.sleep(5)), the engine employs the _gaussian_sleep function. This uses a Normal (Gaussian) distribution to calculate wait times, ensuring that while most delays cluster around a "human-like" average, there are occasional outliers (very short or very long pauses), mirroring real-world human timing.
🔍 4. Feature Breakdown: Beyond Simple Clicks
⌨️ Human-Like Typing Algorithm
The _human_type function is designed to bypass keystroke dynamics analysis. It doesn't just randomize speed; it simulates cognitive errors:

Passport5011, [06/01/2026 01:42 ص]
* Mistake Simulation: The bot has a 9% probability of hitting an adjacent key.
 * Real-time Correction: Upon making an error, it pauses, hits backspace, and types the correct character, exactly as a human would.
🎲 Zero-Value Actions (The Noise Generator)
The engine performs "Zero-Value Actions" via _zero_value_action to create a realistic footprint:
 * Hovering: Moving the mouse over random images or profiles without clicking.
 * Comment Peeking: Opening a comment section, scrolling, and then closing it via the ESCAPE key.
 * Abandonment: Navigating to a group page and leaving without taking any action to simulate "change of mind".
🌍 Multilingual Support & Semantic Targeting
The engine is built for a global scale. It uses a dictionary-based approach to identify UI elements (buttons like "Join," "Like," or "Post") across multiple languages, including Arabic, English, French, and Chinese.
🛡️ 5. Security and Anti-Fingerprinting
To interface with the browser, the project utilizes Undetected Chromedriver (UC). This specialized wrapper modifies the ChromeDriver binary to hide the "CDC" strings and other properties that websites use to detect Selenium.
🚨 Safety Loops and Kill Switches
The engine constantly monitors the page_source for "Security Checkpoints" or "Temporary Blocks". If a threat is detected, the _safety_check function triggers an immediate LONG_SLEEP state (10,800 seconds) and shuts down the driver to protect account integrity.
🖥️ 6. Technical Implementation: The GUI
The project features a high-performance GUI built with customtkinter:
 * Asynchronous Threading: The automation engine runs on a background thread, ensuring the UI remains responsive.
 * Real-Time Logging: A JetBrains Mono-styled log box displays the bot's internal thoughts and state transitions in real-time.
 * Dynamic Configuration: Users can adjust keywords, action budgets, and Chrome profiles without touching the source code.
📝 7. Project Status & Research Note
> Note: This project was developed as an experimental Proof of Concept (PoC) to explore behavioral emulation. While the engine implements advanced human-like patterns, it serves primarily as a technical showcase of complex automation logic and HCI (Human-Computer Interaction) research.
> 
🛠 Installation
 * Clone the Repo: git clone ...
 * Install Dependencies:
   pip install customtkinter selenium undetected-chromedriver numpy

 * Run: python main.py
