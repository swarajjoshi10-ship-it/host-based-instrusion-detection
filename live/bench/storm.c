/* Syscall-heavy micro-benchmark (worst case for any syscall monitor): 5 system calls per iteration. */
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	int n = argc > 1 ? atoi(argv[1]) : 10000;
	char buf[64];
	struct stat st;

	for (int i = 0; i < n; i++) {
		int fd = open("/etc/hostname", O_RDONLY);
		if (read(fd, buf, sizeof(buf)) < 0)
			return 1;
		close(fd);
		stat("/etc/hostname", &st);
		getppid();
	}
	return 0;
}
