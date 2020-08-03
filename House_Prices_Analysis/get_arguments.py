# import for the CLI
import sys
import getopt
import os


# return a list of the following type:
# boolean, boolean, boolean, string
# representing the arguments {automatic, display_plot, save_plots, input_file}
# the default will be returned if no arguments is passed
#
#
# input_file arguments is mandatory
#
# Arguments:
# the defaults are taken from the main script, usually they are:
# automatic = True         |  No input is expected from the user
# display_plot = False     |  Don't display the plots
# save_plots = False       |  Don't save the plots
# dml = False              |  Don't compute the machine learning on the train data to test
# input_file = ''          |  Input file path is set to null and will raise an exception if parsed this way
#
def get_arguments(automatic, display_plot, save_plots, machine_learning, input_file):
    full_cmd_arguments = sys.argv

    argument_list = full_cmd_arguments[1:]

    short_options = 'hudsmi:'
    long_options = ['help', 'user', 'display', 'save','dml', 'input=']

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        sys.exit(2)

    print(arguments)
    return_list = []
    # parse the arguments
    for current_argument, current_value in arguments:
        if current_argument in ('-h', '--help'):  # help argument is parsed and display information
            print('\n\nStart the program in the following order, where [-c] is optional')
            print('python Houses_prices.py [-u] [-d] [-s] -i location_input\n')
            print('-u / --automatic | user interaction with the cmd is required')
            print('-d / --display   | display the plots')
            print('-s / --save      | save the plots in local /Images')
            print('-m / --save      | show the result of ML Algorithm performing with itself')
            print('-i / --input     | input of the data frame\n\n')
            exit(0)
        elif current_argument in ('-u', '--user'):  # user interaction argument is parsed and set the boolean variable
            automatic = False
        elif current_argument in ('-d', '--display'):  # display plots argument is parsed and set the boolean variable
            display_plot = True
        elif current_argument in ('-s', '--save'):  # save plots argument is parsed and set the boolean variable
            save_plots = True
            # create the folder if it doesn't exist
            if not os.path.exists('Images'):  # if the folder 'Images' doesn't exist, create one
                os.makedirs('Images')
        elif current_argument in ('-m', '--dml'):  # user interaction argument is parsed and set the boolean variable
            machine_learning = True
        elif current_argument in ('-i', '--input'):  # input path argument is parsed
            while input_file == '':  # while the input is null, search for it
                input_file = sys.argv[-1]
                if input_file[-4:] != '.csv':  # if the last 4 chars of the command are not good, raise error
                    input_file = ''
                    print('Wrong input, please see the -h/ --help information')
                    exit(0)

    return_list.append(automatic)
    return_list.append(display_plot)
    return_list.append(save_plots)
    return_list.append(machine_learning)
    return_list.append(input_file)
    return return_list
