#include "Server.h"
#define DEVCOUNT 8

int main(){
    Server* server = NewGPUServer();
    server->Initialize(DEVCOUNT);
    server->Run();
    server->Finalize();
}