# ======================
#      SANITY CHECKED
# ======================
# NOTE: If you see red import errors in the editor, select the correct Python interpreter:
# In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> Choose the interpreter where you installed packages (pip install customtkinter selenium undetected-chromedriver numpy)
# The code will still run if packages are installed, even if the editor shows red lines.

# 1. All relevant imports at the very top

import sys
import os
import json
import time
import random
import math
import logging
import threading
import queue

# 1a. Selenium and dependencies (must install all in your environment)
import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException

import numpy as np
from enum import Enum, auto

# 1b. tkinter/CTk and CTkMessagebox
import customtkinter as ctk  # `pip install customtkinter`
import tkinter as tk

# CTkMessagebox should come from customtkinter, not tkinter (pip install customtkinter)
try:
    from customtkinter import CTkMessagebox
except ImportError:
    # fallback so app will still run, but popup is replaced with messagebox
    import tkinter.messagebox as CTkMessagebox

# PIL/Image is not used in this code, so no import needed

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")  # Choose blue or green

# Ensure log path is always absolute for real-time logs
LOG_PATH = os.path.abspath("activity_stream.log")
HISTORY_PATH = os.path.abspath("history.json")

# Session Mood and State Definitions
class SessionMood(Enum):
    LAZY = auto()
    CURIOUS = auto()
    SOCIAL = auto()
    PRODUCTIVE = auto()

class State(Enum):
    IDLING = auto()
    BROWSING = auto()
    DISTRACTED = auto()
    SEARCHING = auto()
    DECIDING = auto()
    EXITING_ABRUPTLY = auto()
    LONG_SLEEP = auto()

class HumanizedFB:
    # --- Path & Data Config -------
    HISTORY_PATH = HISTORY_PATH
    LOG_PATH = LOG_PATH
    MOOD_WEIGHTS = [0.31, 0.31, 0.25, 0.13]
    JOIN_LABELS = ["Join", "انضمام", "Rejoindre", "加入", "تسجيل الانضمام"]
    LIKE_LABELS = ["Like", "أعجبني", "J’aime", "いいね！"]
    COMMENT_LABELS = ["View more comments", "عرض المزيد من التعليقات", "Voir plus de commentaires"]
    POST_LABELS = ["Post", "Publier", "نشر", "Publicar"]
    GROUP_QUEST_Q = [
        "Skip", "Continue", "التالي", "تخطي", "Submit", "Agree", "إرسال", "موافق", "Poursuivre", "Sauter", "Accepter"
    ]
    STATE_SLEEP_MEAN = {
        State.IDLING: 7.0,
        State.BROWSING: 2.5,
        State.DISTRACTED: 4.2,
        State.SEARCHING: 3,
        State.DECIDING: 2,
        State.EXITING_ABRUPTLY: 1,
        State.LONG_SLEEP: 10800
    }
    ABORT_PROB = 0.095

    # 2. Consistent argument names from GUI to bot
    def __init__(
        self, 
        user_data_dir,
        profile_dir,
        keywords,
        post_content,
        enable_joining,
        enable_posting,
        interact_my_groups,
        max_actions
    ):
        # Ensure absolute path to avoid "directory not found" errors
        self.USER_DATA_DIR = os.path.abspath(os.path.expanduser(user_data_dir)) if user_data_dir else None
        self.PROFILE_DIR = profile_dir
        
        # 3. Split safe!
        self.keywords = [k.strip() for k in keywords.split(",") if k.strip()] if isinstance(keywords, str) else (keywords or [])
        self.post_content = post_content
        self.enable_joining = enable_joining
        self.enable_posting = enable_posting
        self.interact_my_groups = interact_my_groups
        self.max_actions = max_actions
        self.driver = None
        self.should_stop = False
        self.exit_flag = False
        self.logger = None
        self.session_start_time = time.time()
        self.state = State.IDLING
        self.mood = None
        self.history = self._load_history()
        self._logger_ready = False

    def _init_logger(self):
        logger = logging.getLogger(f"activity_stream_{id(self)}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.LOG_PATH, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s  %(message)s')
        handler.setFormatter(formatter)
        # Ensure no dupes
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        self._logger_ready = True
        return logger

    def _log(self, msg):
        if not self._logger_ready:
            self.logger = self._init_logger()
        elapsed = int(time.time() - self.session_start_time)
        pattern = f"[{self.mood.name if self.mood else '-'}] [{self.state.name}], spent={elapsed}s :: {msg}"
        self.logger.info(pattern)

    def _launch_driver(self):
        options = uc.ChromeOptions()
        options.add_argument(f'--user-data-dir={self.USER_DATA_DIR}')
        options.add_argument(f'--profile-directory={self.PROFILE_DIR}')
        options.add_argument('--lang=en-US')
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        driver = uc.Chrome(options=options)
        driver.set_window_size(1128, 780)
        return driver

    def _load_history(self):
        try:
            if os.path.exists(self.HISTORY_PATH):
                with open(self.HISTORY_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'visited_urls': [], 'session_patterns': []}

    def _save_history(self):
        try:
            with open(self.HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _gaussian_sleep(self, mean=1.7, std=0.7, min_v=0.4, max_v=7.3):
        if self.should_stop:
            raise Exception("Stopped by user.")
        t = float(np.random.normal(mean, std))
        t = max(min_v, min(max_v, t))
        self._log(f"Sleeping for {round(t,2)}s")
        for _ in range(int(t * 10)):
            if self.should_stop:
                raise Exception("Stopped by user.")
            time.sleep(0.1)
        rem = t-int(t)
        if not self.should_stop:
            time.sleep(rem)

    def _choose_mood(self):
        mood = random.choices(
            [SessionMood.LAZY, SessionMood.CURIOUS, SessionMood.SOCIAL, SessionMood.PRODUCTIVE],
            weights=self.MOOD_WEIGHTS
        )[0]
        self._log(f"Session mood selected: {mood.name}")
        return mood

    def _abort_without_reason(self):
        if random.random() < self.ABORT_PROB:
            self.state = State.EXITING_ABRUPTLY
            self._log("[FSM] Exiting mid-task: boredom/random abort")
            self.exit_flag = True
            return True
        return False

    def _safety_check(self):
        try:
            page_source = self.driver.page_source
            criticals = ["Security Check", "Temporary Block", "Restriction", "Check", "التحقق من الأمان"]
            if any(s in page_source for s in criticals):
                self._log("SAFETY: Detected block/security/restriction page. Entering LONG_SLEEP/KILL.")
                self.state = State.LONG_SLEEP
                self.exit_flag = True
                try:
                    self.driver.quit()
                except Exception:
                    pass
                sys.exit(55)
        except Exception as e:
            self._log(f"SAFETY: Exception {str(e)}")
            pass

    def _find_elements_multilingual(self, label_list, tag_priority=None, partial=True):
        found = []
        tags = tag_priority or ["button", "a", "div", "span"]
        for label in label_list:
            tries = []
            tries += [f"//*[@aria-label='{label}']", f"//*[@title='{label}']"]
            tries += [f"//*[contains(@aria-label, '{label}')]", f"//*[contains(@title, '{label}')]"]
            for t in tags:
                tries.append(f"//{t}[normalize-space(text())='{label}']")
            if partial:
                for t in tags:
                    tries.append(f"//{t}[contains(normalize-space(text()), '{label}')]")
            for xp in tries:
                try:
                    els = self.driver.find_elements(By.XPATH, xp)
                    for el in els:
                        if el.is_displayed():
                            found.append(el)
                except Exception:
                    continue
        dedup = []
        seen = set()
        for el in found:
            try:
                k = el.get_attribute("id") or str(id(el))
                if k and k not in seen:
                    dedup.append(el)
                    seen.add(k)
            except Exception:
                continue
        return dedup

    def _human_type(self, element, text):
        i = 0
        while i < len(text):
            if random.random() < 0.09:
                wrong = random.choice("abcdefghijklmnopqrstuvwxyz")
                element.send_keys(wrong)
                self._gaussian_sleep(0.09, 0.03, 0.05, 0.24)
                element.send_keys(Keys.BACKSPACE)
            element.send_keys(text[i])
            self._gaussian_sleep(0.11, 0.07, 0.065, 0.22)
            i += 1

    def _bezier_mouse_move(self, start, end, steps=None):
        if not steps:
            steps = random.randint(27, 57)
        control1 = (start[0] + random.randint(-40, 60), start[1] + random.randint(32, 90))
        control2 = (end[0] + random.randint(-42, 53), end[1] + random.randint(-75, 25))
        def bezier(t):
            return (
                (1-t)**3*start[0] + 3*(1-t)**2*t*control1[0] + 3*(1-t)*t**2*control2[0] + t**3*end[0],
                (1-t)**3*start[1] + 3*(1-t)**2*t*control1[1] + 3*(1-t)*t**2*control2[1] + t**3*end[1]
            )
        prev = start
        actions = ActionChains(self.driver)
        actions.move_by_offset(prev[0], prev[1])
        for step in range(1, steps+1):
            t = step / steps
            current = bezier(t)
            x_off, y_off = current[0] - prev[0], current[1] - prev[1]
            if abs(x_off) >= 1 or abs(y_off) >= 1:
                try:
                    actions.move_by_offset(x_off, y_off)
                except Exception:
                    pass
                prev = current
            self._gaussian_sleep(0.012, 0.009, 0.006, 0.037)
        try:
            actions.perform()
        except WebDriverException:
            pass

    def _random_micro_scroll(self):
        scroll_amt = random.randint(10, 71)
        direction = random.choice([1, -1])
        body = self.driver.find_element(By.TAG_NAME, 'body')
        for _ in range(random.randint(1, 3)):
            body.send_keys(Keys.ARROW_DOWN if direction > 0 else Keys.ARROW_UP)
            self._gaussian_sleep(0.054, 0.017, 0.03, 0.13)

    def _random_fast_scroll(self):
        scroll_amt = random.randint(180, 400)
        body = self.driver.find_element(By.TAG_NAME, 'body')
        for _ in range(random.randint(5, 9)):
            body.send_keys(Keys.PAGE_DOWN)
            self._gaussian_sleep(0.23, 0.05, 0.1, 0.43)

    def _zero_value_action(self):
        acts = [
            self._hover_random_image_or_name,
            self._open_close_random_comment,
            self._open_group_and_leave
        ]
        act = random.choice(acts)
        self._log(f"[ZERO] Performing zero-value: {act.__name__}")
        act()

    def _hover_random_image_or_name(self):
        options = []
        options += self.driver.find_elements(By.TAG_NAME, "img")
        options += self.driver.find_elements(By.TAG_NAME, "a")
        vis = [el for el in options if el.is_displayed()]
        if vis:
            el = random.choice(vis)
            loc = el.location
            size = el.size
            sx, sy = random.randint(33, 400), random.randint(33, 400)
            ex, ey = loc['x'] + size['width']//2, loc['y'] + size['height']//2
            self._bezier_mouse_move((sx, sy), (ex, ey))
            self._gaussian_sleep(1.2, 0.4, 0.51, 2.4)

    def _open_close_random_comment(self):
        btns = self._find_elements_multilingual(self.COMMENT_LABELS)
        if btns:
            btn = btns[0]
            try:
                btn.click()
                self._gaussian_sleep(1.0, 0.15, 0.4, 2.7)
                for _ in range(random.randint(1, 5)):
                    self._random_micro_scroll()
                self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
                self._gaussian_sleep(0.42, 0.13, 0.23, 1.3)
            except Exception:
                pass

    def _open_group_and_leave(self):
        group_links = self.driver.find_elements(By.XPATH, "//a[contains(@href,'/groups/')]")
        maybes = [g for g in group_links if g.is_displayed()]
        if maybes:
            el = random.choice(maybes)
            loc = el.location
            size = el.size
            sx, sy = random.randint(33, 700), random.randint(33, 700)
            ex, ey = loc['x'] + size['width']//2, loc['y'] + size['height']//2
            self._bezier_mouse_move((sx, sy), (ex, ey))
            self._gaussian_sleep(0.8, 0.26, 0.21, 1.71)
            try:
                el.click()
                self._log("[ZERO] Clicked random group, intent: abandonment!")
                self._gaussian_sleep(random.uniform(9,38), 5, 8, 40)
                for _ in range(random.randint(1,4)):
                    self._random_micro_scroll()
                self.driver.back()
            except Exception:
                pass

    def _handle_group_questionnaire(self):
        timeout = 13.5
        start_time = time.time()
        success = False
        tried_labels = self.GROUP_QUEST_Q
        while (time.time() - start_time) < timeout and not self.should_stop:
            try:
                btns = self._find_elements_multilingual(tried_labels)
                if btns:
                    for btn in btns:
                        if btn.is_displayed() and btn.is_enabled():
                            try:
                                btn.click()
                                self._gaussian_sleep(0.6)
                                self._log("[INTERSTITIAL] Clicked questionnaire button.")
                                success = True
                                break
                            except Exception:
                                continue
                    if success:
                        break
                time.sleep(0.65)
            except Exception:
                pass
        if not success:
            try:
                body = self.driver.find_element(By.TAG_NAME, 'body')
                body.send_keys(Keys.ESCAPE)
                self._log("[INTERSTITIAL] Popup closed via ESCAPE.")
                self._gaussian_sleep(0.7, 0.19, 0.22, 1.33)
            except Exception:
                pass

    def _visit_joinable_groups(self):
        if not self.enable_joining:
            return
        join_btns = self._find_elements_multilingual(self.JOIN_LABELS)
        joined_cnt = 0
        for btn in join_btns:
            if not self.enable_joining:
                break
            if random.random() < 0.37:
                self._log(f"[JOIN] Skipped a join candidate (probabilistic human-like hesitancy).")
                continue
            try:
                start = (random.randint(88, 710), random.randint(77, 700))
                end = (btn.location['x'] + btn.size['width']//2, btn.location['y'] + btn.size['height']//2)
                self._bezier_mouse_move(start, end)
                self._gaussian_sleep(1.1, 0.23, 0.45, 2.33)
                btn.click()
                self._gaussian_sleep(0.9, 0.26, 0.21, 1.54)
                if random.random() < 0.82:
                    self._handle_group_questionnaire()
                joined_cnt += 1
                self._log(f"[JOIN] Attempted group join (count={joined_cnt})")
                if self.max_actions and joined_cnt >= self.max_actions:
                    self._log("Max group join actions for session hit, breaking.")
                    break
            except Exception as e:
                self._log(f"Exception while joining group: {str(e)}")
                continue
            if self._abort_without_reason() or self.should_stop: break

    def _do_social_actions(self):
        if random.random() < 0.18 and self.enable_posting:
            like_btns = self._find_elements_multilingual(self.LIKE_LABELS)
            if like_btns:
                btn = random.choice(like_btns)
                try:
                    start = (random.randint(55, 400), random.randint(22, 211))
                    end = (btn.location['x'] + btn.size['width']//2, btn.location['y'] + btn.size['height']//2)
                    self._bezier_mouse_move(start, end)
                    self._gaussian_sleep(0.33,0.06,0.15,1.08)
                    btn.click()
                    self._log("[SOCIAL] Liked something")
                except Exception:
                    pass
        for _ in range(random.randint(1, 6)):
            self._random_micro_scroll()
        for _ in range(random.randint(0, 2)):
            self._random_fast_scroll()

    def search_groups(self, keyword):
        url = f"https://facebook.com/search/groups/?q={keyword}"
        self.driver.get(url)
        self._log(f"[FLOW] Navigated to group search for '{keyword}'")
        self._gaussian_sleep(3.2, 0.7, 1.15, 6.2)

    def post_text(self, element, text):
        self._human_type(element, text)
        if random.random() < 0.22:
            element.send_keys(Keys.ENTER)
        self._log(f"[POST] Performed humanized text entry of len={len(text)}.")

    def _random_browse_action(self):
        if self.enable_joining and random.random() < 0.40:
            self._visit_joinable_groups()
        else:
            self._zero_value_action()

    def _post_random_to_box(self):
        if not self.enable_posting:
            return
        boxes = self._find_elements_multilingual(self.POST_LABELS)
        candidates = []
        for box in boxes:
            try:
                if box.is_enabled() and box.is_displayed():
                    candidates.append(box)
            except Exception:
                continue
        if candidates:
            content = self.post_content if self.post_content.strip() else random.choice([
                "😂", "Wow!", "Cool Info!", "مثير للاهتمام!", "Merci pour le partage!", "🔥"
            ])
            try:
                self.post_text(random.choice(candidates), content)
            except Exception:
                pass

    def _do_my_groups_interactions(self):
        # stub for extra behavior if you want to interact with user's joined groups in future
        self._log("[GROUPS] Option to interact w/ my groups enabled (not yet implemented).")

    def run(self):
        self.should_stop = False
        try:
            self.driver = self._launch_driver()
        except Exception as e:
            self._log(f"Chrome launch error: {e}")
            return
        self.mood = self._choose_mood()
        self.state = State.BROWSING
        last_pattern = []
        action_counter = 0

        try:
            self.driver.get("https://facebook.com/")
            self._gaussian_sleep(2.2, 0.3, 0.7, 4.0)

            while (not self.exit_flag) and (not self.should_stop) and (not self.max_actions or action_counter < self.max_actions):
                self._safety_check()
                cur_url = self.driver.current_url
                if cur_url not in self.history['visited_urls'][-12:]:
                    self.history['visited_urls'].append(cur_url)
                last_pattern.append(self.state.name)
                time_spent = int(time.time()-self.session_start_time)
                self._log(f"Pat:{self.mood.name} -> State:{self.state.name} at {cur_url}, t={time_spent}")

                self._save_history()
                if self.mood == SessionMood.LAZY:
                    self.state = State.IDLING
                    self._gaussian_sleep(self.STATE_SLEEP_MEAN[self.state], 1.1, 1.7, 12)
                    if random.random() < 0.3:
                        self._random_browse_action()
                        action_counter += 1
                    else:
                        self._do_social_actions()
                elif self.mood == SessionMood.CURIOUS:
                    self.state = State.SEARCHING
                    if random.random() < 0.65 and self.keywords:
                        kwrd = random.choice(self.keywords)
                        self.search_groups(kwrd)
                    self._gaussian_sleep(self.STATE_SLEEP_MEAN[self.state], 0.8, 0.8, 6)
                    self._random_browse_action()
                    action_counter += 1
                elif self.mood == SessionMood.SOCIAL:
                    self.state = State.BROWSING
                    self._do_social_actions()
                    self._gaussian_sleep(self.STATE_SLEEP_MEAN[self.state], 0.8, 0.7, 8)
                    if random.random() < 0.14:
                        self._post_random_to_box()
                        action_counter += 1
                elif self.mood == SessionMood.PRODUCTIVE:
                    self.state = State.DECIDING
                    if random.random() < 0.72:
                        if random.random() < 0.48 and self.keywords:
                            kwrd = random.choice(self.keywords)
                            self.search_groups(kwrd)
                            self._visit_joinable_groups()
                            action_counter += 1
                        else:
                            self._visit_joinable_groups()
                            action_counter += 1
                    if random.random() < 0.23:
                        self._post_random_to_box()
                        action_counter += 1
                    self._gaussian_sleep(self.STATE_SLEEP_MEAN[self.state], 0.9, 0.7, 7)
                if self.interact_my_groups:
                    self._do_my_groups_interactions()
                if random.random() < 0.19:
                    prev = self.mood
                    self.mood = self._choose_mood()
                    self._log(f"[MOOD DRIFT] {prev.name} -> {self.mood.name}")
                if random.random() < 0.06:
                    self.state = State.DISTRACTED
                    self._log("[FSM] Randomly distracted (scrolling/surfing).")
                    self._gaussian_sleep(self.STATE_SLEEP_MEAN[self.state], 0.6, 0.4, 15)
                    self._zero_value_action()
                if random.random() < 0.031:
                    self.state = State.EXITING_ABRUPTLY
                    self._log("[Exit] Exiting session abruptly for human-like unpredictability.")
                    break
                if self._abort_without_reason(): break
        except KeyboardInterrupt:
            self._log("KeyboardInterrupt (SIGINT) detected.")
        except Exception as e:
            self._log(f"Exception: {e}")
        finally:
            self._save_history()
            try:
                if self.driver:
                    self.driver.quit()
                    self._log("Driver quit, session end.")
            except Exception:
                pass

    def emergency_stop(self):
        self.should_stop = True
        self.exit_flag = True
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass

# ---------------------- GUI SECTION ----------------------

class BotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HumanizedFB Automation Bot [Dark Mode]")
        self.geometry("1150x620")
        self.resizable(False, False)
        self._bot_thread = None
        self._bot_instance = None
        self._log_queue = queue.Queue()
        self._stop_refresh = False

        # ---- Left Panel (Settings) ----
        left_panel = ctk.CTkFrame(master=self, width=270, fg_color="#181b1e")
        left_panel.pack(side="left", fill="y", padx=8, pady=8)

        ctk.CTkLabel(left_panel, text="FB Settings", font=("Segoe UI", 20, "bold")).pack(pady=(12,5))

        self.keywords_entry = ctk.CTkEntry(left_panel, height=32, width=200, placeholder_text="Keywords (comma separated)")
        self.keywords_entry.pack(pady=(8, 2))
        self.keywords_entry.insert(0, "chatgpt, AI, productivity, بزنس")

        self.post_content_entry = ctk.CTkEntry(left_panel, height=32, width=200, placeholder_text="Post Content")
        self.post_content_entry.pack(pady=(6, 2))
        self.post_content_entry.insert(0, "🚀 Automating Facebook (for research)")

        ctk.CTkLabel(left_panel, text="Chrome User Data Dir", font=("Segoe UI", 13)).pack(pady=(10,0))
        self.profile_path_entry = ctk.CTkEntry(left_panel, height=32, width=235)
        self.profile_path_entry.pack()
        self.profile_path_entry.insert(0, r"C:\Users\hp\AppData\Local\Google\Chrome\User Data")

        ctk.CTkLabel(left_panel, text="Profile Name", font=("Segoe UI", 13)).pack(pady=(10,0))
        self.profile_name_entry = ctk.CTkEntry(left_panel, height=32, width=235)
        self.profile_name_entry.pack()
        self.profile_name_entry.insert(0, "Profile 4")

        # ---- Center Panel (Controls) ----
        center_panel = ctk.CTkFrame(self, fg_color="#242c36", width=315)
        center_panel.pack(side="left", fill="y", padx=(2,8), pady=8)
        ctk.CTkLabel(center_panel, text="Automation Controls", font=("Segoe UI",20,"bold")).pack(pady=(12,7))

        self.toggle_join = ctk.CTkSwitch(center_panel, text="Enable Group Joining", switch_height=24)
        self.toggle_join.pack(pady=(12, 7))
        self.toggle_join.select()

        self.toggle_post = ctk.CTkSwitch(center_panel, text="Enable Posting", switch_height=24)
        self.toggle_post.pack(pady=(7,7))
        self.toggle_post.select()

        self.toggle_interact = ctk.CTkSwitch(center_panel, text="Interact With My Groups (N/A)", switch_height=24)
        self.toggle_interact.pack(pady=(7,22))
        self.toggle_interact.configure(state="disabled")

        ctk.CTkLabel(center_panel, text="Max Actions Per Session", font=("Segoe UI", 13)).pack(pady=(2,0))
        self.max_actions_slider = ctk.CTkSlider(center_panel, from_=1, to=50, width=220)
        self.max_actions_slider.set(8)
        self.max_actions_slider.pack(pady=(2,8))

        self.max_actions_label = ctk.CTkLabel(center_panel, text="8")
        self.max_actions_label.pack()
        self.max_actions_slider.configure(command=lambda v: self.max_actions_label.configure(text=f"{int(float(v))}"))

        button_colors = {"fg_color": "#11A8FF", "hover_color": "#088ed3"}
        self.start_button = ctk.CTkButton(center_panel, text="START BOT", font=("Segoe UI",18,"bold"), height=45, width=190, **button_colors, command=self.start_bot)
        self.start_button.pack(pady=(18,5))
        self.stop_button = ctk.CTkButton(center_panel, text="EMERGENCY STOP", font=("Segoe UI",16,"bold"), height=36, width=168, fg_color="#fa383e", hover_color="#dc2441", command=self.stop_bot)
        self.stop_button.pack(pady=6)

        # ---- Right Panel (Live Logs) ----
        right_panel = ctk.CTkFrame(self, fg_color="#0E1013")
        right_panel.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        ctk.CTkLabel(right_panel, text="Live Session Log", font=("Segoe UI",20,"bold")).pack(anchor="w", padx=10, pady=(13,6))

        self.log_box = ctk.CTkTextbox(right_panel, width=470, height=532, font=("JetBrains Mono",12), fg_color="#181b1e", text_color="#e6eefe", wrap='word')
        self.log_box.pack(padx=8, pady=5, fill="both", expand=True)

        self.log_box.configure(state="disabled")
        self.after(1000, self.refresh_log_textbox)
        self.protocol("WM_DELETE_WINDOW", self.gui_close)

        self.configure(bg="#21232b")
        left_panel.configure(border_width=0)
        center_panel.configure(border_width=0)
        right_panel.configure(border_width=0)

    def start_bot(self):
        if self._bot_thread and self._bot_thread.is_alive():
            try:
                CTkMessagebox(title="Already Running", message="The bot is already running.", icon="warning")
            except Exception:
                tk.messagebox.showwarning("Already Running", "The bot is already running.")  # fallback
            return
        self.log_box.configure(state='normal')
        self.log_box.delete(1.0, tk.END)
        self.log_box.configure(state='disabled')

        # Get all GUI entries and pass directly to HumanizedFB
        # Normalize user_data_dir to absolute path to avoid "directory not found" errors
        user_data_dir = os.path.abspath(os.path.expanduser(self.profile_path_entry.get().strip()))
        profile_dir = self.profile_name_entry.get().strip()
        keywords = self.keywords_entry.get().strip()
        content = self.post_content_entry.get()
        enable_joining = bool(self.toggle_join.get())
        enable_posting = bool(self.toggle_post.get())
        interact_my_groups = False
        max_actions = int(float(self.max_actions_slider.get()))

        self._bot_instance = HumanizedFB(
            user_data_dir=user_data_dir,
            profile_dir=profile_dir,
            keywords=keywords,
            post_content=content,
            enable_joining=enable_joining,
            enable_posting=enable_posting,
            interact_my_groups=interact_my_groups,
            max_actions=max_actions
        )
        self._bot_thread = threading.Thread(target=self._bot_instance.run, daemon=True)
        self._stop_refresh = False
        self._bot_thread.start()

    def stop_bot(self):
        if self._bot_instance:
            self._bot_instance.emergency_stop()
        self._stop_refresh = True

    def refresh_log_textbox(self):
        log_path = HumanizedFB.LOG_PATH
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
        except Exception:
            lines = []
        current_text = self.log_box.get(1.0, tk.END)
        content = "".join(lines)
        if content != current_text:
            self.log_box.configure(state='normal')
            self.log_box.delete(1.0, tk.END)
            self.log_box.insert(tk.END, content)
            self.log_box.see(tk.END)
            self.log_box.configure(state='disabled')
        if not self._stop_refresh:
            self.after(1600, self.refresh_log_textbox)

    def gui_close(self):
        self.stop_bot()
        self.destroy()

if __name__ == "__main__":
    app = BotGUI()
    app.mainloop()
    