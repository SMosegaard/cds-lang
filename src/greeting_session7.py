import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",
                        "-n",
                        required = True)
    parser.add_argument("--likes",
                        "-l",
                        required = True)
    args = parser.parse_args()
    return args

class Person:

    def __init__(self, name):
        self.name = name
        self.likes = likes

    def hello(self):
        print("Hello, " + self.name)

    def preferences(self):
        print(f"I like " + self.likes + "!")

# a function inside a class = method
# good practise to name a class with upper case

def main():
    args = parser()
    person = Person(args.name, args.likes)
    person.hello()
    person.preferences()

if __name__=="__main__":
    main()

