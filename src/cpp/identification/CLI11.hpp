#pragma once
#include <string>
#include <vector>
#include <stdexcept>

namespace CLI {
class Error : public std::runtime_error {
public:
    explicit Error(const std::string &msg) : std::runtime_error(msg) {}
};

class Option {
public:
    std::string name;
    bool required{false};
    std::vector<std::string> *vec_target{nullptr};
    std::string *str_target{nullptr};
    int *int_target{nullptr};
    double *dbl_target{nullptr};
};

class App {
    std::vector<Option> opts;
public:
    App(const std::string &description="") {}

    Option &add_option(const std::string &name, std::string &var, const std::string &desc="") {
        Option o; o.name=name; o.str_target=&var; opts.push_back(o); return opts.back();
    }
    Option &add_option(const std::string &name, std::vector<std::string> &var, const std::string &desc="") {
        Option o; o.name=name; o.vec_target=&var; opts.push_back(o); return opts.back();
    }
    Option &add_option(const std::string &name, int &var, const std::string &desc="") {
        Option o; o.name=name; o.int_target=&var; opts.push_back(o); return opts.back();
    }
    Option &add_option(const std::string &name, double &var, const std::string &desc="") {
        Option o; o.name=name; o.dbl_target=&var; opts.push_back(o); return opts.back();
    }

    void parse(int argc, char **argv) {
        for (int i=1; i<argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0) {
                std::string key = arg.substr(2);
                Option *opt = nullptr;
                for (auto &o : opts) {
                    if (o.name == key) { opt = &o; break; }
                }
                if (!opt) throw Error("Unknown option: " + arg);
                if (opt->vec_target) {
                    while (i+1 < argc && std::string(argv[i+1]).rfind("--",0) != 0) {
                        opt->vec_target->push_back(argv[++i]);
                    }
                    if (opt->vec_target->empty()) throw Error("Option requires values: " + arg);
                } else if (opt->str_target) {
                    if (i+1 >= argc) throw Error("Option requires value: " + arg);
                    *opt->str_target = argv[++i];
                } else if (opt->int_target) {
                    if (i+1 >= argc) throw Error("Option requires value: " + arg);
                    *opt->int_target = std::stoi(argv[++i]);
                } else if (opt->dbl_target) {
                    if (i+1 >= argc) throw Error("Option requires value: " + arg);
                    *opt->dbl_target = std::stod(argv[++i]);
                }
            } else {
                throw Error("Unknown argument: " + arg);
            }
        }
        for (auto &o : opts) {
            bool provided = (o.vec_target && !o.vec_target->empty()) ||
                            (o.str_target && !o.str_target->empty()) ||
                            (o.int_target) || (o.dbl_target);
            if (o.required && !provided) throw Error("Missing required option: --" + o.name);
        }
    }
};
} // namespace CLI