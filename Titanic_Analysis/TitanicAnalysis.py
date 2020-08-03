# The normal imports
import os
import pandas as pd

import VisualizationModule
import MachineLearningModule


########################################################################################################################
# Functions that are useful and needed
# A classic clear screen function that works on any OS
def clear_screen():
    # clear screen
    if os.name == 'nt':
        os.system('cls')  # on windows, clear screen
    else:
        os.system('clear')


# Wait for the input of the user and then clear the screen
def wait_input_clear_screen_action():
    input("Press any key to continue!")
    clear_screen()


# Wait for the input of the user and then clear the screen
def wait_input_no_action():
    input("Press any key to continue!")


########################################################################################################################

# import the CVS into the DataFrame
titanic_DFrame = pd.read_csv('train.csv')
# turn off the warnings on cmd
pd.options.mode.chained_assignment = None  # default='warn'

clear_screen()

# set the while condition to true, in order to enter into the program
input_option = 1
while True:
    clear_screen()
    print('''First some basic questions:\n
    1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc) \n
    2.) What deck were the passengers on and how does that relate to their class?\n
    3.) Where did the passengers come from?\n
    4.) Who was alone and who was with family?\n
    5.) What factors helped someone survive the sinking? \n
    6.) Machine output of a similar input \n\n
    7.) Exit \n\n
    ''')

    while True:
        try:
            input_option = int(input('Which do you want to be answered? '))
            break
        except:
            print("Invalid, try again!")

    titanic_module = VisualizationModule.VisualizationClass()
    machine_module = MachineLearningModule.Machine_Learning_Class()
    clear_screen()
    if input_option == 1:
        # Interpret and display plots concerning some general information about the passengers
        titanic_module.general_plots(titanic_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 2:
        # Interpret and display plots concerning the deck level and changes of survival
        titanic_module.deck_plots(titanic_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 3:
        # Interpret and display plots concerning the embarking location and changes of survival
        titanic_module.embarking_plots(titanic_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 4:
        # Interpret and display plots concerning the family status on board and changes of survival
        titanic_module.family_alone_plots(titanic_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 5:
        # Interpret and display plots concerning all the important factors and changes of survival
        titanic_module.survival_factors_plots(titanic_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 6:
        # import the CVS into the DataFrame
        test_DFrame = pd.read_csv('test.csv')
        # Interpret and make predictions on the survivability of a new data set
        machine_module.machine_learning(titanic_DFrame, test_DFrame)
        wait_input_clear_screen_action()
    elif input_option == 7:
        break
    else:
        continue

clear_screen()
