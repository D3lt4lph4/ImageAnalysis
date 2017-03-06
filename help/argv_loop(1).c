// Example : loop through list of command line arguments (files)

// Author: Toby Breckon, toby.breckon@cranfield.ac.uk
// License : GPL, http://www.gnu.org/copyleft/gpl.html
// Version : 0.11

#include <stdlib.h>
#include <stdio.h>

// N.B. To make this work correctly with MS Visual Studio 2003/05/08/10+ 
// do the following:
// * Open Project Menu -> <your project name> Properties
// * Open Configuration Properties -> Linker Tab -> Input 
// * In the "Additional Dependancies" box (where you also have the OpenCV dependancies)
//   add in the filename "setargv.obj" (with no quotes, after the OpenCV stuff that is  
//   already there)
//
// When you compile you may get a warning about there not being a debug library found
// or similar. Ignore this. Run program of the form "programename /path/to/files/*" and
// the file names in the specified directory should be correctly displayed by the following
// code. No change to the C/C++ code is required.
//
// Reference : http://msdn2.microsoft.com/en-us/library/8bch7bkk.aspx

// N.B. #2 This only works correctly with wildcards (i.e. /path/to/files/*jpg) when 
// there are no spaces in the file path. Use relative paths from the executable location
// to the required files.

int main(int argc, char *argv[])
{
  int i;

    for(i = 1; i < argc; i++)
    {
      printf("File %i : %s\n", i, argv[i]);
    }
}
