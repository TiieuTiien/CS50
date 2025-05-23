import copy
import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        
        domainCopy = copy.deepcopy(self.domains)
        
        for v in domainCopy:
            length = v.length
            for x in domainCopy[v]:
                if len(x) != length:
                    self.domains[v].remove(x)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        
        # return overlap cell of x and y
        xoverlap, yoverlap = self.crossword.overlaps[x, y]
        
        # make a domains copy
        domainsCopy = copy.deepcopy(self.domains)
        
        if xoverlap:
            
            for xword in domainsCopy[x]:
                wordMatched = False
                
                for yword in self.domains[y]:
                    # if x and y have the same word at the overlap position
                    if xword[xoverlap] == yword[yoverlap]:
                        wordMatched = True
                        break
                
                if wordMatched:
                    continue
                else:
                    revised = True
                    self.domains[x].remove(xword)
                    
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = []
        
        if arcs != None:
            queue = arcs
        else:
            for var1 in self.domains:
                for var2 in self.crossword.neighbors(var1):
                    if self.crossword.overlaps[var1, var2] is not None:
                        queue.append((var1, var2))
                        
        while len(queue) > 0:
            # DEQUEUE queue
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x):
                    # ENQUEUE queue
                    queue.append((z, x))
                    
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.domains:
            if variable not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # all values are distinct
        distinct = [*assignment.values()]
        if len(distinct) != len(set(distinct)):
            return False 
        
        for keys in assignment:
            # every value is the correct length
            if len(assignment[keys]) != keys.length:
                return False

            # and there are no conflicts between neighboring variables.
            for keys2 in self.crossword.neighbors(keys):
                if keys2 in assignment:
                    x, y = self.crossword.overlaps[keys, keys2]
                    if assignment[keys][x] != assignment[keys2][y]:
                        return False
        
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Create empty dict
        wordDict = {}

        # loop through var
        for word in self.domains[var]:
            # don't count neighbors in the assignment
            if word in assignment: 
                continue
            else:
                # create order
                order = 0
                
                # if word inside neighbor, increase order
                for neighbor in self.crossword.neighbors(var):
                    if word in neighbor:
                        order += 1
                            
                # assign order to wordDict
                wordDict[word] = order
            
        # sort the dict by order
        return sorted(wordDict, key=lambda key: wordDict[key])
        
    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # create unassigned dict
        unassignedDict = {}
        
        for variable in self.domains:
            if variable not in assignment:
                # if variable not in assignment, add it to unassignedDict
                unassignedDict[variable] = self.domains[variable]
                
        # return first variable in the dict sorted by number of remaing values
        return sorted(unassignedDict, key=lambda key: unassignedDict[key])[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # if assignment is complete return
        if len(assignment) == len(self.domains):
            return assignment
        
        # select unassigned variable
        unassigned = self.select_unassigned_variable(assignment)
        
        for word in self.domains[unassigned]:
            # make an assignment copy and update variable word
            assignmentCopy = assignment.copy()
            assignmentCopy[unassigned] = word
            # check for consistency
            if self.consistent(assignmentCopy):
                result = self.backtrack(assignmentCopy)
                if result is not None:
                    return result
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
