from model import inference, train, test, checkUp

print("------------------------------------------")
print("1. Train")
print("2. Inference")
print("3. Test")
print("4. CheckUp")
print("------------------------------------------")
choice = input("What u wanna do (When they come for you.. Bad Boys.. )\n")

if choice == "1":
    train()

if choice == "2":
    inference()

if choice == "3":
    test()

if choice == "4":
    checkUp()