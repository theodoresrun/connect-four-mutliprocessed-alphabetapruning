#IMPORTED NUMPY TO CREATE ARRAYS
#IMPORTED OS TO USE CLEAR FUCNTION
#IMPORTED TIME TO HANDLE ANIMATONS
#IMPORTED RANDOM FOR CHOOSING A STARTING POSITION
import numpy as np, os, time, random, multiprocessing as mp
global calculations
calculations = 0
finaloppvalues = []
turn = 0
depth = 8

def clear_console(): #USES OS TO CLEAR THE CONSOLE SCREEN, TO MAINTAIN SIMPLE AESTHETIC
    os.system('cls' if os.name == 'nt' else 'clear')


def define_array(): #CREATES AN INTEGER 6X7 ARRAY 'MAIN MATRIX' THAT FUNCTIONS AS THE MAIN 'DATA' REFERENCE. RETURNS THE MAIN MATRIX
    matrix = np.zeros((6,7), dtype='int')
    return matrix


def visualize(matrix): #CREATES A STRING-BASED 6X7 ARRAY 'VISUAL MATRIX' THAT TAKES THE MAIN ARRAY AND CONVERTS IT INTO USABLE VISUAL INFORMATION. RETURNS THE VISUAL MATRIX
    visualmatrix = np.zeros((6,7), dtype='str')
    rowcount = 0
    colcount = 0
    for columns in visualmatrix:
        colcount = 0
        for rows in columns:
            if matrix[rowcount,colcount] == 0:
                visualmatrix[rowcount,colcount] = ' '
            elif matrix[rowcount,colcount] == 1:
                visualmatrix[rowcount,colcount] = 'O'
            elif matrix[rowcount,colcount] == -1:
                visualmatrix[rowcount,colcount] = 'X'
            colcount += 1
        rowcount += 1
    return visualmatrix
#Format is ROW,COLUMN


def board_output(visualmatrix): #OUTPUTS THE VISUAL MATRIX TO THE CONSOLE, AND MAKES IT PRETTY
    print(f"Current Depth = {depth}")
    print("|1|2|3|4|5|6|7|")
    for i in range(6):
        #Code optimization, using nested loops was too slow.
        print(f"|{visualmatrix[i,0]}|{visualmatrix[i,1]}|{visualmatrix[i,2]}|{visualmatrix[i,3]}|{visualmatrix[i,4]}|{visualmatrix[i,5]}|{visualmatrix[i,6]}|")
    print()

    
def pindrop(matrix): #CHECKS THE MAIN MATRIX, AND FINDS IF ANY TOKENS CAN BE 'DROPPED' BY A TILE. IF A TOKEN HAS AN EMPTY SPACE BELOW IT, IT WILL COPY THE TOKEN TO THE TILE BELOW, AND ERASE THE CURRENT TILE. ONLY 'DROPS' BY ONE ROW. RETURNS THE NEW MAIN MATRIX.
    for i in range(6,-1,-1):
        for j in range(7):
            if i != 6:
                if matrix[i,j] != 0:
                    try:
                        if matrix[i+1,j] == 0:
                            matrix[i+1,j] = matrix[i,j]
                            matrix[i,j] = 0
                    except:
                        continue
    return matrix

#DEFINE THE DIFFERENT MUTLIPROCESSING BRANCHES TO ALLOW FOR PARALLEL PROCESSING - 7 BRANCHES FOR 7 POSSIBILITIES.
def branch_1(mat, finaloppvalues, calc_counter):
    i = 1
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_2(mat, finaloppvalues, calc_counter):
    i = 2
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_3(mat, finaloppvalues, calc_counter):
    i = 3
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_4(mat, finaloppvalues, calc_counter):
    i = 4
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_5(mat, finaloppvalues, calc_counter):
    i = 5
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_6(mat, finaloppvalues, calc_counter):
    i = 6
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)

def branch_7(mat, finaloppvalues, calc_counter):
    i = 7
    temp_mat = mat.copy()
    temp_mat[0, (i-1)] = -1
    for m in range(7):
        temp_mat = pindrop(temp_mat)
    currentvalue = best_move_player(temp_mat, calc_counter)
    finaloppvalues.append(currentvalue)


def drop_all(matrix):
    """Ensure tokens fall to the bottom (apply pindrop until stable)."""
    new_matrix = matrix.copy()
    while True:
        old = new_matrix.copy()
        new_matrix = pindrop(new_matrix)
        if np.array_equal(old, new_matrix):
            break
    return new_matrix

def win_checker(matrix):
    win = False
    winner = ''
    # Check rows.
    for i in range(6):
        for j in range(4):
            summation = matrix[i, j] + matrix[i, j+1] + matrix[i, j+2] + matrix[i, j+3]
            if summation == 4:
                winner = "Player 1"
                win = True
            elif summation == -4:
                winner = "Player 2"
                win = True
    # Check columns.
    for i in range(3):
        for j in range(7):
            summation = matrix[i, j] + matrix[i+1, j] + matrix[i+2, j] + matrix[i+3, j]
            if summation == 4:
                winner = "Player 1"
                win = True
            elif summation == -4:
                winner = "Player 2"
                win = True
    # Check diagonals (down-right).
    for i in range(3):
        for j in range(4):
            summation = matrix[i, j] + matrix[i+1, j+1] + matrix[i+2, j+2] + matrix[i+3, j+3]
            if summation == 4:
                winner = "Player 1"
                win = True
            elif summation == -4:
                winner = "Player 2"
                win = True
    # Check diagonals (down-left).
    for i in range(3):
        for j in range(3, 7):
            summation = matrix[i, j] + matrix[i+1, j-1] + matrix[i+2, j-2] + matrix[i+3, j-3]
            if summation == 4:
                winner = "Player 1"
                win = True
            elif summation == -4:
                winner = "Player 2"
                win = True
    return win, winner

def value_checker(matrix):
    """A heuristic function that returns an evaluation.
       Replace with your own evaluation logic as needed."""
    value = 0
    # Scan rows.
    for i in range(6):
        for j in range(4):
            current = matrix[i, j] + matrix[i, j+1] + matrix[i, j+2] + matrix[i, j+3]
            if current == 4:
                value += 100000
            elif current == -4:
                value -= 100000
            elif current == 3:
                value += 1000
            elif current == -3:
                value -= 1000
    # Scan columns.
    for i in range(3):
        for j in range(7):
            current = matrix[i, j] + matrix[i+1, j] + matrix[i+2, j] + matrix[i+3, j]
            if current == 4:
                value += 100000
            elif current == -4:
                value -= 100000
            elif current == 3:
                value += 1000
            elif current == -3:
                value -= 1000
    # Diagonals (down-right).
    for i in range(3):
        for j in range(4):
            current = matrix[i, j] + matrix[i+1, j+1] + matrix[i+2, j+2] + matrix[i+3, j+3]
            if current == 4:
                value += 100000
            elif current == -4:
                value -= 100000
            elif current == 3:
                value += 1000
            elif current == -3:
                value -= 1000
    # Diagonals (down-left).
    for i in range(3):
        for j in range(3, 7):
            current = matrix[i, j] + matrix[i+1, j-1] + matrix[i+2, j-2] + matrix[i+3, j-3]
            if current == 4:
                value += 100000
            elif current == -4:
                value -= 100000
            elif current == 3:
                value += 1000
            elif current == -3:
                value -= 1000
    return value

def available_moves(matrix):
    """Return a list of available column indices (0-based) where a move can be made."""
    return [col for col in range(7) if matrix[0, col] == 0]

# ---------------------------
# Alpha-beta minimax with pruning

def alphabeta(matrix, depth, alpha, beta, maximizingPlayer, calc_counter):
    # Terminal test: depth 0 or game over.
    win, _ = win_checker(matrix)
    if depth == 0 or win:
        return value_checker(matrix)
    
    moves = available_moves(matrix)
    if not moves:
        return value_checker(matrix)
    
    if maximizingPlayer:
        value = -float('inf')
        for move in moves:
            new_matrix = matrix.copy()
            # Maximizing turn uses Player 1 (token +1).
            new_matrix[0, move] = 1
            new_matrix = drop_all(new_matrix)
            score = alphabeta(new_matrix, depth - 1, alpha, beta, False, calc_counter)
            value = max(value, score)
            alpha = max(alpha, value)
            with calc_counter.get_lock():
                calc_counter.value += 1
                print(f"Calculations done: {calc_counter.value}", end="\r")
            if beta <= alpha:
                break  # Beta cutoff.
        return value
    else:
        value = float('inf')
        for move in moves:
            new_matrix = matrix.copy()
            # Minimizing turn uses Player 2 (token -1).
            new_matrix[0, move] = -1
            new_matrix = drop_all(new_matrix)
            score = alphabeta(new_matrix, depth - 1, alpha, beta, True, calc_counter)
            value = min(value, score)
            beta = min(beta, value)
            with calc_counter.get_lock():
                calc_counter.value += 1
                print(f"Calculations done: {calc_counter.value}", end="\r")
            if beta <= alpha:
                break  # Alpha cutoff.
        return value

# ---------------------------
# Parallel branch function using alpha-beta

def alpha_branch(mat, move, depth, calc_counter, result_list):
    """Evaluate one candidate move in parallel.
       'move' is the column index (0-based) where AI places its token (-1)."""
    new_matrix = mat.copy()
    new_matrix[0, move] = -1  # AI move.
    new_matrix = drop_all(new_matrix)
    # After AI moves, it becomes the opponent’s turn (maximizing).
    score = alphabeta(new_matrix, depth - 1, -float('inf'), float('inf'), True, calc_counter)
    # Append tuple (move, score) to the shared result list.
    result_list.append((move, score))

# ---------------------------
# AI solver using parallel processing

def AI_solver(matrix, depth):
    # Create shared objects: a Manager list for branch results and a shared counter.
    manager = mp.Manager()
    result_list = manager.list()
    calc_counter = mp.Value('i', 0)
    
    moves = available_moves(matrix)
    processes = []
    
    # For each available move, start a process to evaluate it.
    for move in moves:
        p = mp.Process(target=alpha_branch, args=(matrix, move, depth, calc_counter, result_list))
        processes.append(p)
        p.start()
        time.sleep(0.15)  
    
    for p in processes:
        p.join()
    
    # Now select the move with the minimum score (better for the AI playing as -1).
    # (If you prefer the opposite convention, change the selection accordingly.)
    best_move = None
    best_score = float('inf')
    for move, score in result_list:
        if score < best_score:
            best_score = score
            best_move = move
    print(f"Calculations done: {calc_counter.value}")
    time.sleep(2)
    # Return the chosen move as a 1-indexed column.
    return best_move + 1


 
def value_checker(matrix): #ASSIGNS A VALUE FOR A POSITION ON THE BOARD. I'D LOVE FOR THIS TO BE MORE COMPLEX AND ITERATIVE, BUT THIS HAS TO BE CALLED A LOT, AND NEEDS TO BE EFFICIENTLY READ
#First Check for predictive Wins or Losses
    #rows, 4 scans per row
    def line_checker(i,j,info,matrix): #FUNCTION ONLY CALLED THE LEAST AMOUNT OF TIMES TO SAVE ON COMPUTE
        if info == 'row':
            return matrix[i,j] + matrix[i,j+1] + matrix[i,j+2] + matrix[i,j+3]
        elif info == 'column':
            return matrix[i,j] + matrix[i+1,j] + matrix[i+2,j] + matrix[i+3,j]
        elif info == 'diagonalupright':
            return matrix[i,j] + matrix[i+1,j+1] + matrix[i+2,j+2] + matrix[i+3,j+3]
        elif info == 'diagonalupleft':
            return matrix[i,j] + matrix[i+1,j-1] + matrix[i+2,j-2] + matrix[i+3,j-3]
            
    value = 0
    for i in range(6):
        for j in range(4):
            currentvalues = line_checker(i,j,'row',matrix)
            if currentvalues == 4:
                value += 100000
            elif currentvalues == -4:
                value -= 100000
            if currentvalues == 3:
                value += 1000
            elif currentvalues == -3:
                value -= 1000   
    #columns, 3 scans per column
    for i in range(3):
        for j in range(6):
            currentvalues = line_checker(i,j,'column',matrix)
            if currentvalues == 4:
                value += 100000
            if currentvalues == -4:
                value -= 100000
            if currentvalues == 3:
                value += 1000
            if currentvalues == -3:
                value -= 1000   
    #diagonals, so help me god
    for i in range(3):
        for j in range(4):
            currentvalues = line_checker(i,j,'diagonalupright',matrix)
            if currentvalues == 4:
                value += 100000
            elif currentvalues == -4:
                value -= 100000
            if currentvalues == 3:
                value += 1000
            elif currentvalues == -3:
                value -= 1000   
    for i in range(3):
        for j in range(4):
            currentvalues = line_checker(i,j,'diagonalupleft',matrix)
            if currentvalues == 4:
                value += 100000
            elif currentvalues == -4:
                value -= 100000
            if currentvalues == 3:
                value += 1000
            elif currentvalues == -3:
                value -= 1000   
    return value
def win_checker(matrix): #FUNCTION CHECKS WHETHER OR NOT THE BOARD CONTAINS A WINNING MOVE. RETURNS A BOOLEAN AND WHO THE WINNER IS. 
    win = False
    winner = ''
    #rows, 4 scans per row
    for i in range(6):
        for j in range(4):
            if matrix[i,j] + matrix[i,j+1] + matrix[i,j+2] + matrix[i,j+3] == 4:
                winner = "Player 1"
                win = True
            elif matrix[i,j] + matrix[i,j+1] + matrix[i,j+2] + matrix[i,j+3] == -4:
                winner = "Player 2"
                win = True
    #columns, 3 scans per column
    for i in range(3):
        for j in range(6):
            if matrix[i,j] + matrix[i+1,j] + matrix[i+2,j] + matrix[i+3,j] == 4:
                winner = "Player 1"
                win = True
            elif matrix[i,j] + matrix[i+1,j] + matrix[i+2,j] + matrix[i+3,j] == -4:
                winner = "Player 2"
                win = True
    #diagonals, so help me god
    for i in range(3):
        for j in range(4):
            if matrix[i,j] + matrix[i+1,j+1] + matrix[i+2,j+2] + matrix[i+3,j+3] == 4:
                winner = "Player 1"
                win = True
            elif matrix[i,j] + matrix[i+1,j+1] + matrix[i+2,j+2] + matrix[i+3,j+3] == -4:
                winner = "Player 2"
                win = True
    for i in range(3):
        for j in range(3,7):
            if matrix[i,j] + matrix[i+1,j-1] + matrix[i+2,j-2] + matrix[i+3,j-3] == 4:
                winner = "Player 1"
                win = True
            elif matrix[i,j] + matrix[i+1,j-1] + matrix[i+2,j-2] + matrix[i+3,j-3] == -4:
                winner = "Player 2"
                win = True
    return win, winner
if __name__ == '__main__': #NECESSARY DUE TO MULTIPROCESSING
    while True:
        for o in range(4):
            clear_console()
            time.sleep(0.5)
            print(" ████  ███  █   █ █   █ █████  ████ █████    █████  ███  █   █ ████  \n█     █   █ ██  █ ██  █ █     █       █      █     █   █ █   █ █   █ \n█     █   █ █ █ █ █ █ █ ████  █       █      ████  █   █ █   █ ████  \n█     █   █ █  ██ █  ██ █     █       █      █     █   █ █   █ █   █ \n ████  ███  █   █ █   █ █████  ████   █      █      ███  █████ █   █ ")
            time.sleep(0.5)
        matrix = define_array()
        '''
        while True:
            clear_console()
            try:
                depth = int(input("What algorithm difficulty would you like to play at? "))
            except:
                print("That's not a valid number")
                time.sleep(2)
            else:
                if depth < 1 or depth > 20:
                    print("That's not a valid number")
                else:
                    break
        '''
        
        
        turn = 0
        
        playermove = ''
        while True:
            for i in range(6): #DROPS THE GIVEN TOKENS DOWN
                visualmatrix = visualize(matrix)
                board_output(visualmatrix)
                matrix = pindrop(matrix)
                time.sleep(0.1)
                clear_console()
            clear_console()
            board_output(visualmatrix)
            win, winner = win_checker(matrix)
            if win == True:
                for i in range(5):
                    clear_console()
                    time.sleep(0.3)
                    print("█   █ █████ █   █ █   █ █████ ████  \n█   █   █   ██  █ ██  █ █     █   █ \n█ █ █   █   █ █ █ █ █ █ ████  ████  \n██ ██   █   █  ██ █  ██ █     █   █ \n█   █ █████ █   █ █   █ █████ █   █ ")
                    time.sleep(0.2)
                time.sleep(0.7)
                clear_console()
                if winner == "Player 1":
                    print("████  █      ███  █   █ █████ ████     ███  █   █ █████ \n█   █ █     █   █  █ █  █     █   █   █   █ ██  █ █     \n████  █     █████   █   ████  ████    █   █ █ █ █ ████  \n█     █     █   █   █   █     █   █   █   █ █  ██ █     \n█     █████ █   █   █   █████ █   █    ███  █   █ █████ ")
                elif winner == "Player 2":
                    print(" ███  █████    █   █ █████ █   █ █████ \n█   █   █      █   █   █   ██  █ █     \n█████   █      █ █ █   █   █ █ █ █████ \n█   █   █      ██ ██   █   █  ██     █ \n█   █ █████    █   █ █████ █   █ █████ ")
                break

            while True:
                if turn > 10:
                    depth = 5 + (turn // 3)
                else:
                    depth = 8
                if turn % 2 == 0:
                    playermove = "Player 1"
                    move = input(f"Enter your move, with an integer 1~7, or 'end': ")
                elif turn % 2 != 0:
                    move = AI_solver(matrix, depth)
                    
                if move == 'end':
                    exit()
                else:
                    try:
                        move = int(move)
                    except:
                        print("Put a number in, dum dum.")
                        continue
                if move <= 0 or move > 7:
                    print("That's not a move dumbass")
                else:
                    break
            #turn switching
            if turn % 2 == 0:
                matrix[0,(move-1)] = 1
            elif turn % 2 != 0:
                matrix[0,(move)-1] = -1
            turn += 1
            clear_console()

        while True:
            board_output(visualmatrix)
            playagain = input("Play Again? (Y/N): ")
            if str.lower(playagain) == 'y':
                break
            elif str.lower(playagain) == 'n':
                exit()
            else:
                print("Please say (Y/N)...")
