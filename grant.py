import matplotlib.pyplot as plt

# Define tasks, start weeks, and durations
tasks = [
    {"name": "Project Planning & Requirements Gathering", "start": 1, "duration": 1},
    {"name": "System Design (Model Selection & Data Preprocessing)", "start": 2, "duration": 2},
    {"name": "Model Development & Training", "start": 4, "duration": 4},
    {"name": "Model Evaluation & Testing", "start": 8, "duration": 2},
    {"name": "Deployment & Reporting", "start": 10, "duration": 1},
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Generate bars for each task
for i, task in enumerate(tasks):
    ax.broken_barh([(task["start"], task["duration"])], 
                   (i - 0.4, 0.8), facecolors="lightgray", edgecolor="black")

# Add labels for tasks
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task["name"] for task in tasks])

# Format the chart
ax.set_xlabel("Weeks")
ax.set_ylabel("Tasks")
ax.set_xlim(0, 12)  # Adjust based on the project timeline
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("Project Timeline (Gantt Chart)", fontsize=14, fontweight="bold")

# Improve layout
plt.tight_layout()

# Show the chart
plt.show()
