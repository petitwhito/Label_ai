import pandas as pd

# Read the CSV file
df = pd.read_csv("../dataset/job_descriptions.csv")

# Define the replacement function
def replace_qualification(qualification):
    if qualification[0] == 'B':
        return 1
    elif qualification[0] == 'M':
        return 2
    else:
        return 0  # Keep the original value if it doesn't match the conditions

# Apply the replacement function to the "Qualifications" column
df["Qualifications"] = df["Qualifications"].apply(replace_qualification)

# Keep only the "Job Description" and "Qualifications" columns
df = df[["Job Description", "Qualifications"]]

# Save the modified DataFrame to a new CSV file
df.to_csv("../dataset/updated_job_descriptions.csv", index=False)

print("Replacement completed. Check the output file: output.csv")