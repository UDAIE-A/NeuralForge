#!/usr/bin/env python3
"""
Generate a large conversational dataset for training NeuralForge
to have better day-to-day conversations.
"""

import random
import os

# Define conversation templates and examples
greetings = [
    "Hello! How can I help you today?",
    "Hi there! What brings you here?",
    "Hey! Nice to see you. What's on your mind?",
    "Good morning! How are you feeling today?",
    "Good afternoon! I hope you're having a great day.",
    "Good evening! Is there anything I can help you with?",
    "Hey there! I'm glad you stopped by.",
    "Hi! I'm here if you need anything.",
    "Hello! Ready to chat?",
    "Welcome! How can I assist you today?"
]

feelings_response = [
    "I'm here to listen. What's been on your mind?",
    "That sounds tough. I'm here if you want to talk about it.",
    "I'm sorry to hear that. Sometimes it helps to talk things through.",
    "It's okay to feel that way. Would you like some suggestions?",
    "I understand. Let's see if we can work through this together.",
    "That must be difficult. I'm here to help however I can.",
    "Take a deep breath. We can figure this out together.",
    "It's normal to feel that way. You're not alone.",
    "I'm sorry you're going through this. Is there something specific bothering you?",
    "Let's focus on finding a solution. What would help you feel better?"
]

how_are_you = [
    "I'm doing well, thank you for asking! How about yourself?",
    "I'm great! How are you feeling today?",
    "Doing pretty good. It's a lovely day, isn't it?",
    "I'm fine, thanks. What's new with you?",
    "Can't complain! What's been going on in your world?",
    "I'm good! Just here, ready to help. And you?",
    "All good here. Tell me about your day!",
    "Pretty well, thanks. What can I do for you today?",
    "I'm doing alright. How has your day been so far?",
    "Great, thanks for asking! What brings you here today?"
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
    ("Are you more of a morning or a night person?", "Definitely a morning person. I love the quiet hours of the day. How about you?")]

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
    ("How do I find my passion?", "Try new things! Your passion isn't something you find by sitting still. Pay attention to what makes you lose track of time and what you're naturally curious about.")
]

fun_facts = [
    "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
    "Here's a fun fact: Bananas are berries, but strawberries are not! Botanically speaking, a berry is a fruit produced from the ovary of a single flower.",
    "Octopuses have three hearts, and their blood is blue! Two hearts pump blood to the gills, and one pumps it to the rest of the body.",
    "A group of flamingos is called a 'flamboyance'. Isn't that fitting?",
    "Did you know that a day on Venus is longer than a year on Venus? It takes Venus longer to rotate once onMEMORY on its axis than to orbit the sun.",
    "Wombat poop is cube-shaped. This helps it stay in place on rocks and marks their territory.",
    "There are more stars in the universe than grains of sand on all the Earth's beaches. It's a truly mind-boggling number!",
    "A jiffy is an actual unit of time. It is 1/100th of a second.",
    "Did you know that the inventor of the frisbee was turned into a frisbee? Walter Morrison's ashes were molded into a frisbee after he died.",
    "The Eiffel Tower can actually be 15 cm taller during the summer. This is because of thermal expansion; the iron heats up and expands."
]

jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my computer I needed a break, and it said 'No problem, I have plenty of backups.'",  
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "I'm reading a book about anti-gravity. It's impossible to put down!",
    "What do you call a fake noodle? An impasta!",
    "Why did the math book look sad? Because it had too many problems.",
    "I would tell you a joke about an elevator, but it's an uplifting experience.",
    "Why don't skeletons fight each other? They don't have the guts.",
    "What do you call a fish with no eyes? A fsh!",
    "Why did the bicycle fall over? Because it was two tired!"
]

# Generate larger dataset
def generate_dataset(target_lines=50000):
    lines = []
    
    # Add greetings and small talk
    for greeting in greetings:
        lines.append("User:")
        lines.append(greeting)
        lines.append("")
        
    # Add daily conversations
    for q, a in daily_conversations:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
        lines.append("")
        
    # Add Q&A pairs
    for q, a in qa_pairs:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
        lines.append("")
        
    # Add fun facts
    for fact in fun_facts:
        lines.append("User: Tell me something interesting.")
        lines.append(f"Assistant: {fact}")
        lines.append("")
        
    # Add jokes
    for joke in jokes:
        lines.append("User: Tell me a joke.")
        lines.append(f"Assistant: {joke}")
        lines.append("")
        
    # Generate variations with random combinations to reach target
    templates = [
        "User: {0}\nAssistant: {1}\n",
        "User: Can you tell me {0}?\nAssistant: Sure! {1}\n",
        "User: I was wondering, {0}?\nAssistant: That's a great question. {1}\n",
        "User: {0}\nAssistant: {1}\nUser: Can you elaborate?\nAssistant: Of course! {1} I hope that clarifies things.\n",
        "User: {0}\nAssistant: {1}\nUser: Thanks!\nAssistant: You're very welcome! Feel free to ask if you have any other questions.\n",
    ]
    
    all_qa = daily_conversations + qa_pairs
    
    while len(lines) < target_lines:
        q, a = random.choice(all_qa)
        template = random.choice(templates)
        lines.append(template.format(q, a))
        
    return '\n'.join(lines[:target_lines])

if __name__ == "__main__":
    print("Generating conversational dataset...")
    dataset = generate_dataset(target_lines=50000)
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "conversational_train_large.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dataset)
    
    lines = dataset.count('\n')
    print(f"Generated dataset with {lines} lines")
    print(f"Saved to: {output_path}")
