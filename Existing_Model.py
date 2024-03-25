import gender_guesser.detector as gender

# A bit easy, not much to see here

# Initialize gender detector
detector = gender.Detector()

def predict_gender(name):
    return detector.get_gender(name)


while True:
    name = input("Enter your name: ")
    predicted_gender = predict_gender(name)
    print(f"{name} is a {predicted_gender}")
