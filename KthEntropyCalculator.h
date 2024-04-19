#include <iostream>
#include <unordered_map>
#include <cmath>
#include <string>
#include <vector>

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ h2;
    }
};

class KthEntropyCalculator
{
private:
    std::vector<unsigned char> text;
    int k;
    std::unordered_map<std::string, int> k_tuples_count;
    std::unordered_map<std::pair<std::string, unsigned char>, int, pair_hash> followers_count;

public:
    KthEntropyCalculator(std::vector<unsigned char> text, int k) : text(text), k(k)
    {
        k_tuples_and_followers();
    }

    void k_tuples_and_followers()
    {
        for (int i = 0; i < text.size() - k; ++i)
        {
            std::vector<unsigned char> k_tuple(text.begin() + i, text.begin() + i + k);
            char next_char = text[i + k];

            std::string k_tuple_str(k_tuple.begin(), k_tuple.end());
            k_tuples_count[k_tuple_str]++;
            followers_count[{k_tuple_str, next_char}]++;
        }
    }

    double calculate_kth_entropy()
    {
        double entropy = 0;
        double n = text.size();

        for (const auto &entry : followers_count)
        {
            const auto &key = entry.first;
            double f_wx = static_cast<double>(entry.second);
            double f_w = static_cast<double>(k_tuples_count[key.first]);
            if (f_wx != 0 && f_w != 0)
            {
                entropy -= (f_wx / n) * std::log2(f_wx / f_w);
            }
        }

        return entropy;
    }
};