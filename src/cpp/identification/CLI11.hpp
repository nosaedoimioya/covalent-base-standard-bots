#pragma once
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace CLI {

class Error : public std::runtime_error {
    public:
        explicit Error(const std::string &msg) : std::runtime_error(msg) {}
    };
using ParseError = Error;

class Option {
public:
    std::string name;
    bool required{false};
    bool seen{false};
    bool positional{false};

    std::vector<std::string> *vec_target{nullptr};
    std::string *str_target{nullptr};
    int *int_target{nullptr};
    double *dbl_target{nullptr};
    bool *bool_target{nullptr};

    // simple string validator (used by IsMember)
    std::function<void(const std::string &)> str_validator{};

    // allow "opt->check(IsMember(...))"
    Option &check(const std::function<void(const std::string &)> &fn) {
        str_validator = fn;
        return *this;
    }
};

inline std::function<void(const std::string &)>
IsMember(const std::vector<std::string> &allowed) {
    return [allowed](const std::string &v) {
        if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
            std::ostringstream oss;
            oss << "Invalid value '" << v << "'. Allowed: ";
            for (size_t i = 0; i < allowed.size(); ++i) {
                if (i) oss << ", ";
                oss << allowed[i];
            }
            throw Error(oss.str());
        }
    };
}

inline std::function<void(const std::string &)>
IsMember(std::initializer_list<const char *> allowed_init) {
    std::vector<std::string> allowed;
    allowed.reserve(allowed_init.size());
    for (auto *p : allowed_init) allowed.emplace_back(p);
    return IsMember(allowed);
}

class App {
    std::vector<Option> opts;
    std::vector<size_t> positional_indices;
public:
    App(const std::string &description = "") { (void)description; }

    Option &add_option(const std::string &name, std::string &var, const std::string &desc = "") {
        (void)desc;
        Option o; o.name = name; o.str_target = &var;
        if (name.rfind("--", 0) != 0) o.positional = true;
        opts.push_back(o);
        if (opts.back().positional) positional_indices.push_back(opts.size() - 1);
        return opts.back();
    }
    Option &add_option(const std::string &name, std::vector<std::string> &var, const std::string &desc = "") {
        (void)desc;
        Option o; o.name = name; o.vec_target = &var;
        if (name.rfind("--", 0) != 0) o.positional = true;
        opts.push_back(o);
        if (opts.back().positional) positional_indices.push_back(opts.size() - 1);
        return opts.back();
    }
    Option &add_option(const std::string &name, int &var, const std::string &desc = "") {
        (void)desc;
        Option o; o.name = name; o.int_target = &var;
        if (name.rfind("--", 0) != 0) o.positional = true;
        opts.push_back(o);
        if (opts.back().positional) positional_indices.push_back(opts.size() - 1);
        return opts.back();
    }
    Option &add_option(const std::string &name, double &var, const std::string &desc = "") {
        (void)desc;
        Option o; o.name = name; o.dbl_target = &var;
        if (name.rfind("--", 0) != 0) o.positional = true;
        opts.push_back(o);
        if (opts.back().positional) positional_indices.push_back(opts.size() - 1);
        return opts.back();
    }
    Option &add_flag(const std::string &name, bool &var, const std::string &desc = "") {
        (void)desc;
        Option o; o.name = name; o.bool_target = &var;
        // flags are always prefixed ("--saved-maps") so not positional
        opts.push_back(o);
        return opts.back();
    }

    void parse(int argc, char **argv) {
        size_t next_positional = 0;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg.rfind("--", 0) == 0) {
                // named option / flag
                std::string key = arg.substr(2);
                Option *opt = nullptr;
                for (auto &o : opts) {
                    if (o.name == key) { opt = &o; break; }
                }
                if (!opt) throw Error("Unknown option: " + arg);
                opt->seen = true;

                if (opt->vec_target) {
                    while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                        opt->vec_target->push_back(argv[++i]);
                    }
                    if (opt->vec_target->empty())
                        throw Error("Option requires values: " + arg);
                } else if (opt->str_target) {
                    if (i + 1 >= argc) throw Error("Option requires value: " + arg);
                    std::string v = argv[++i];
                    if (opt->str_validator) opt->str_validator(v);
                    *opt->str_target = std::move(v);
                } else if (opt->int_target) {
                    if (i + 1 >= argc) throw Error("Option requires value: " + arg);
                    *opt->int_target = std::stoi(argv[++i]);
                } else if (opt->dbl_target) {
                    if (i + 1 >= argc) throw Error("Option requires value: " + arg);
                    *opt->dbl_target = std::stod(argv[++i]);
                } else if (opt->bool_target) {
                    *opt->bool_target = true;
                }
            } else {
                // positional argument
                if (next_positional >= positional_indices.size())
                    throw Error("Unexpected positional argument: " + arg);
                Option &opt = opts[positional_indices[next_positional++]];
                opt.seen = true;
                if (opt.str_target) {
                    std::string v = arg;
                    if (opt.str_validator) opt.str_validator(v);
                    *opt.str_target = std::move(v);
                } else if (opt.int_target) {
                    *opt.int_target = std::stoi(arg);
                } else if (opt.dbl_target) {
                    *opt.dbl_target = std::stod(arg);
                } else {
                    throw Error("Positional argument bound to unsupported type: " + opt.name);
                }
            }
        }

        for (auto &o : opts) {
            if (o.required && !o.seen) throw Error("Missing required option: --" + o.name);
        }
    }
};

} // namespace CLI
