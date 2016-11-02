#include "utils.h"
#include "dirent.h"

std::vector<std::string> get_files_in_dir(const std::string& dir_path)
{
	std::vector<std::string> result;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(dir_path.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
			{
				continue;
			}
			if (ent->d_type == DT_REG){
				result.push_back(dir_path + ent->d_name);
			}
			else if (ent->d_type == DT_DIR)
			{
				std::vector<std::string> sub_dir_files = get_files_in_dir(dir_path + ent->d_name+"\\");
				std::copy(sub_dir_files.begin(), sub_dir_files.end(), std::back_inserter(result));
			}
		}
		closedir(dir);
	}
	return result;
}
