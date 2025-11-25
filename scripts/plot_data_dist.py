import matplotlib.pyplot as plt
import numpy as np

# Data
labels = [
    "False Information", 
    "Animal Abuse",
    "Environmental Damage",
    "Security Threats",
    "Privacy",
    "Illegal Activities",
    "Academic or Financial Fraud"
]
sizes = [22.2, 6.3, 6.3, 12.4, 10.8, 22.5, 19.6]

# Modern color palette
#colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
colors = ['#A8D5E2', '#87CEEB', '#6CB4DD', '#4A9ECC', '#357ABD', '#2E5F8F', '#5B9BD5']


# Create figure
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Create pie chart
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    textprops={'fontsize': 17, 'weight': 'normal', 'color': '#2C3E50', 'style': 'italic', 'family': 'sans-serif'},
    wedgeprops={'linewidth': 2, 'edgecolor': 'white', 'antialiased': True}
)

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('semibold')
    autotext.set_fontsize(17)

# Title
ax.set_title('Distribution of Data Categories', 
             fontsize=17, 
             weight='bold', 
             color='#2C3E50',
             pad=20)

# Perfect circle
ax.axis('equal')
plt.rcParams["font.family"] = "monospace"  # or "sans-serif", "monospace"
plt.tight_layout()
plt.savefig('data_dist.pdf', bbox_inches='tight')
plt.show()
