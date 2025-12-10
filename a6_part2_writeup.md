# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Bedrooms
2. Bathrooms
3. Age
4. Least Important: Squarefeet

**Explanation:**

I ordered them by which one had the largest effect on pricing. Bedrooms had by far the most impactso it was first and the rest followed. 



---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Each additional bedroom increases the pice by $6648.97

**Feature 2:**
Each additional bathroom increases the price by $3858.90

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**

My R² was 0.9936 which tells me that the model is pretty accurate because it is very close to 1. There is room for improvement because I can still get it closer to 1. 


---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Location

**Why it would help:**
It would help because house are cheaper in less populated enviornments and neighborhoods matter when looking at house pricing. 

**Feature 2:**
Number of floors

**Why it would help:**
It would because typically more expensive houses have more floors and vis versa. 

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**

No because we would be extrapolating the data so it might not be accurate. Our data does not have the ability to predict this because it is too far from our data. 
