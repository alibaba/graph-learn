#include <pthread.h>
#include <semaphore.h>

#ifdef __APPLE__

// Referred and modified from
//
//      https://stackoverflow.com/questions/641126/posix-semaphores-on-mac-os-x-sem-timedwait-alternative

typedef struct
{
    pthread_mutex_t count_lock;
    pthread_cond_t  count_bump;
    unsigned count;
}
bosal_sem_t;

using macos_sem_t = int64_t;

int sem_init(macos_sem_t *psem, int flags, unsigned count);
int sem_destroy(macos_sem_t *psem);
int sem_post(macos_sem_t *psem);
int sem_trywait(macos_sem_t *psem);
int sem_wait(macos_sem_t *psem);
int sem_timedwait(macos_sem_t *psem, const struct timespec *abstim);

#endif
