#include "Server.h"
#include <stdlib.h>

int main(int argc, char** argv){
    Server* server = NewGPUServer();
    server->Initialize(atoi(argv[1]));
    server->Run();
    server->Finalize();
}