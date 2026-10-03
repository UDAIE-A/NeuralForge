#!/usr/bin/env python3
"""
Build a BALANCED conversational corpus: every user prompt gets enough
examples, greetings heavily represented, no prompt allowed to dominate.
"""

import os
import random

random.seed(42)

greeting_words = [
    "hello", "hi", "hey", "hey there", "hi there", "hello there",
    "yo", "greetings", "hiya", "what's up", "good morning", "good afternoon",
    "good evening", "howdy", "how's it going", "nice to see you",
]

greeting_responses = [
    "Hello! How can I help you today?",
    "Hi there! What brings you here?",
    "Hey! Nice to see you. What's on your mind?",
    "Hello! Ready to chat?",
    "Hi! I'm here if you need anything.",
    "Hey there! I'm glad you stopped by.",
    "Hello! Welcome. What would you like to talk about?",
    "Hi! Great to hear from you. What's up?",
    "Hey! How is your day going so far?",
    "Hello to you too! What can I do for you?",
    "Well hello! Good to see you. How can I assist you today?",
    "Hi there! Good to see you. What can I help you with?",
    "Hey! I was hoping someone would say hi. What's on your mind?",
    "Hello! It's nice to talk to you. How can I help?",
    "Hi! I'm all ears. What would you like to chat about?",
    "Hey! How are things on your end?",
]

qa_pairs = [
    ("What is the best way to learn a new skill?", "The best way is through consistent practice and breaking the skill down into smaller, manageable parts. Don't be afraid to make mistakes; they are your best teachers."),
    ("How can I be more productive?", "Start with a clear to-do list and prioritize your most important tasks. Minimize distractions by turning off notifications, and try working in focused intervals, like the Pomodoro Technique."),
    ("What should I do when I'm bored?", "Try picking up a new hobby like drawing, writing, or learning an instrument. Alternatively, you could go for a walk, call a friend, or read a book."),
    ("How do I make new friends?", "Start by joining clubs or groups based on your interests. Be a good listener and show genuine interest in others. Sometimes, just being kind and approachable is all it takes."),
    ("What is a good way to deal with stress?", "Physical activity is a great stress reliever. Also, try mindfulness or meditation. Even just a few minutes of deep breathing can make a big difference."),
    ("How can I improve my memory?", "Get enough sleep, stay physically active, and challenge your brain with puzzles or learning new things. Techniques like creating associations and visualizations can also help."),
    ("What is the secret to happiness?", "There's no single secret, but gratitude and strong relationships are key. Focus on what you have rather than what you lack and spend time with people who uplift you."),
    ("How do I start a conversation with someone I just met?", "A simple compliment or a question about their surroundings is a great icebreaker. People usually love talking about their interests."),
    ("What are some fun activities to do on a rainy day?", "You could have a movie marathon, bake cookies, read a book, or start a new craft project. It's a great excuse to get cozy indoors!"),
    ("How can I become a better listener?", "Focus entirely on the person speaking. Avoid interrupting and try to understand their perspective before responding. Sometimes, just being present is enough."),
    ("What makes a good leader?", "Empathy, clear communication, and the ability to inspire others. A good leader listens to their team and leads by example."),
    ("How do I stay motivated to exercise?", "Find an activity you actually enjoy, set small achievable goals, and track your progress. Having a workout buddy can also make it more fun."),
    ("What's the best way to start the day?", "A good morning routine sets the tone for the day. Try not to check your phone immediately. Instead, stretch, drink water, and have a nutritious breakfast."),
    ("How can I be more creative?", "Expose yourself to new experiences and ideas. Don't be afraid of bad ideas; sometimes they lead to the best ones. Give yourself time to daydream and explore."),
    ("What is a good book for self-improvement?", "Atomic Habits by James Clear is a fantastic choice. It provides practical strategies for building good habits and breaking bad ones."),
    ("How do I handle difficult people?", "Stay calm and don't take their behavior personally. Set boundaries and try to understand where they're coming from. Sometimes, a little empathy goes a long way."),
    ("What are some healthy snacks?", "Fruits like apples and berries, nuts, Greek yogurt, and hummus with veggies are all great options that provide energy without a sugar crash."),
    ("How can I improve my writing?", "Write every day, even if it's just a few sentences. Read widely to see how other authors craft their stories, and don't be afraid to edit and rewrite."),
    ("What is the most important life skill?", "Emotional intelligence. The ability to understand and manage your own emotions, and empathize with others, is invaluable in almost every aspect of life."),
    ("How do I find my passion?", "Try new things! Your passion isn't something you find by sitting still. Pay attention to what makes you lose track of time and what you're naturally curious about."),
]

daily_conversations = [
    ("Hey, what's up?", "Not much, just hanging out. How about you? What have you been up to?"),
    ("How was your weekend?", "It was pretty relaxing. I got to catch up on some reading and spend time with family. How about yours?"),
    ("What do you do for fun?", "I enjoy reading, going for walks, and trying new recipes. It's the simple things that make me happy. What about you?"),
    ("Do you have any plans for today?", "I have a few things on my to-do list, but I want to keep it light. Maybe go for a walk later if the weather is nice. Do you have any plans?"),
    ("What's your favorite way to relax after a long day?", "For me, it's making a nice cup of tea and reading a good book. It really helps me unwind. What about you?"),
    ("Have you seen any good movies lately?", "Yes! I recently watched a really interesting documentary. It was about the deep ocean. What about you? Any recommendations?"),
    ("What kind of music do you listen to?", "I have pretty eclectic taste, but I lean towards indie and acoustic music. It's great for focusing. What do you like?"),
    ("Did you do anything interesting today?", "Not too much out of the ordinary, but I did try a new coffee shop which was fun. How was your day?"),
    ("What do you usually eat for breakfast?", "I usually go for something simple like oatmeal with fruit or a smoothie. It keeps me energized. What about you?"),
    ("Are you more of a morning or a night person?", "Definitely a morning person. I love the quiet hours of the day. How about you?"),
]


def build():
    blocks = []

    # Greetings: every greeting word x every response, repeated enough for the
    # model to learn "hello -> greeting" as strongly as anything else.
    for _ in range(6):
        for word in greeting_words:
            for resp in greeting_responses:
                blocks.append(f"User: {word}\nAssistant: {resp}")

    # Core Q&A: each pair a healthy number of times with varied templates.
    templates = [
        "User: {q}\nAssistant: {a}",
        "User: Can you tell me {q}\nAssistant: Sure! {a}",
        "User: I was wondering, {q}\nAssistant: That's a great question. {a}",
    ]
    for _ in range(60):
        for q, a in qa_pairs:
            t = random.choice(templates)
            blocks.append(t.format(q=q, a=a))

    # Daily conversations: each pair repeated.
    for _ in range(60):
        for q, a in daily_conversations:
            blocks.append(f"User: {q}\nAssistant: {a}")

    # A few follow-up chains so "thanks!" / "elaborate" exist but stay rare.
    for _ in range(10):
        for q, a in qa_pairs:
            blocks.append(f"User: {q}\nAssistant: {a}\nUser: Thanks!\nAssistant: You're very welcome! Feel free to ask if you have any other questions.")
            blocks.append(f"User: {q}\nAssistant: {a}\nUser: Can you elaborate?\nAssistant: Of course! {a}")

    random.shuffle(blocks)
    return "\n\n".join(blocks) + "\n"


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "conversational_train_large.txt")
    corpus = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Balanced corpus: {len(corpus):,} chars, {len(corpus.splitlines())} lines")
