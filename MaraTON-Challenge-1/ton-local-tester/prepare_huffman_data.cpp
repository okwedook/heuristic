#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ext/random>

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC optimize("O3", "unroll-loops")
    #pragma GCC target("sse4.2")
#endif

using namespace std;

using ll = long long;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using graph = vector<vector<int>>;

const ld eps = 1e-9;
const int mod = 1000000007;
const ll inf = 3000000000000000007ll;

#define pb push_back
#define pf push_front
#define popb pop_back
#define popf pop_front
#define f first
#define s second
#define all(a) (a).begin(), (a).end()
#define rall(a) (a).rbegin(), (a).rend()
#define by_key(...) [](const auto &a, const auto &b) { return a.__VA_ARGS__ < b.__VA_ARGS__; }

#ifdef DEBUG
    __gnu_cxx::sfmt19937 gen(857204);
#else
    __gnu_cxx::sfmt19937 gen(int(chrono::high_resolution_clock::now().time_since_epoch().count()));
#endif

template<class T, class U> inline bool chmin(T &x, const U& y) { return y < x ? x = y, 1 : 0; }
template<class T, class U> inline bool chmax(T &x, const U& y) { return y > x ? x = y, 1 : 0; }
template<class T> inline int sz(const T &a) { return a.size(); }
template<class T> inline void sort(T &a) { sort(all(a)); }
template<class T> inline void rsort(T &a) { sort(rall(a)); }
template<class T> inline void reverse(T &a) { reverse(all(a)); }
template<class T> inline T sorted(T a) { sort(a); return a; }
struct InitIO {
    InitIO() {
        ios_base::sync_with_stdio(0);
        cin.tie(0);
        cout.tie(0);
        cout << fixed << setprecision(12);
    }
    ~InitIO() {
        #ifdef DEBUG
            cerr << "Runtime is: " << clock() * 1.0 / CLOCKS_PER_SEC << endl;
        #endif
    }
} Initter;

template<class T, class U> inline istream& operator>>(istream& str, pair<T, U> &p) { return str >> p.f >> p.s; }
template<class T> inline istream& operator>>(istream& str, vector<T> &a) { for (auto &i : a) str >> i; return str; }

void flush() { cout << flush; }
void flushln() { cout << endl; }
template<class T> void print(const T &x) { cout << x; }
template<class T> void read(T &x) { cin >> x; }
template<class T, class ...U> void read(T &x, U&... u) { read(x); read(u...); }
template<class T, class ...U> void print(const T &x, const U&... u) { print(x); print(u...); }
template<class ...T> void println(const T&... u) { print(u..., '\n'); }
#if __cplusplus >= 201703L
    namespace TypeTraits {
        template<class T> constexpr bool IsString = false;
        template<> constexpr bool IsString<string> = true;
        template<class T, class = void> struct IsIterableStruct : false_type{};
        template<class T>
        struct IsIterableStruct<
            T,
            void_t<
                decltype(begin(declval<T>())),
                decltype(end(declval<T>()))
            >
        > : true_type{};
        template<class T> constexpr bool IsIterable = IsIterableStruct<T>::value;
        template<class T> constexpr bool NonStringIterable = !IsString<T> && IsIterable<T>;
        template<class T> constexpr bool DoubleIterable = IsIterable<decltype(*begin(declval<T>()))>;
    };
    // Declaration (for cross-recursion)
    template<class T>
    auto pdbg(const T&x) -> enable_if_t<!TypeTraits::NonStringIterable<T>, string>;
    string pdbg(const string &x);
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<TypeTraits::NonStringIterable<T>, string>;
    template<class T, class U>
    string pdbg(const pair<T, U> &x);

    // Implementation
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<!TypeTraits::NonStringIterable<T>, string> {
        stringstream ss;
        ss << x;
        return ss.str();
    }
    template<class T, class U>
    string pdbg(const pair<T, U> &x) {
        return "{" + pdbg(x.f) + "," + pdbg(x.s) + "}";
    }
    string pdbg(const string &x) {
        return "\"" + x + "\"";
    }
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<TypeTraits::NonStringIterable<T>, string> {
        auto begin = x.begin();
        auto end = x.end();
        string del = "";
        if (TypeTraits::DoubleIterable<T>) {
            del = "\n";
        }
        string ans;
        ans += "{" + del;
        if (begin != end) ans += pdbg(*begin++);
        while (begin != end) {
            ans += "," + del + pdbg(*begin++);
        }
        ans += del + "}";
        return ans;
    }
    template<class T> string dbgout(const T &x) { return pdbg(x); }
    template<class T, class... U>
    string dbgout(T const &t, const U &... u) {
        string ans = pdbg(t);
        ans += ", ";
        ans += dbgout(u...);
        return ans;
    }
#endif

#ifdef DEBUG
    #define dbg(...) print("[", #__VA_ARGS__, "] = ", dbgout(__VA_ARGS__)), flushln()
    #define msg(...) print("[", __VA_ARGS__, "]"), flushln()
#else
    #define dbg(...) 0
    #define msg(...) 0
#endif


using distribution_data = std::vector<std::pair<long long, int>>;

struct HuffmanEncoder {
  HuffmanEncoder() {}
  HuffmanEncoder(const distribution_data& data, const std::string name) {
    set_data(data);
    #ifndef ONLINE_JUDGE
      eval_data(data, name);
    #endif
    build_index();
  }
  void eval_data(const distribution_data& data, const std::string name) {
    long long uncompressed = 0, compressed = 0;
    for (auto [count, value] : data) {
      auto [_, len] = code_len.at(value);
      uncompressed += 8 * count;
      compressed += len * count;
    }
    auto compression_ratio = uncompressed * 1.0 / compressed;
    std::cerr << "Huffman encoder data for " << name << ": " << dbgout(uncompressed, compressed, compression_ratio, data.size()) << '\n';
  }
  int get_len(int value) const {
    return code_len.at(value).second;
  }
  bool have_value(int value) const {
    return code_len.count(value);
  }
protected:
  void set_data(const distribution_data& data) {
    assert(data.size() > 1);
    using vertex = std::pair<long long, std::vector<int>>;
    auto cmp = [&](const vertex& lhs, const vertex& rhs) {
      if (lhs.first != rhs.first) return lhs.first > rhs.first;
      return lhs.second.size() > rhs.second.size();
    };
    std::priority_queue<vertex, std::vector<vertex>, decltype(cmp)> sorted_vertex(cmp);
    for (auto [cnt, value] : data) {
      sorted_vertex.push({cnt, {value}});
    };
    while (sorted_vertex.size() > 1) {
      auto [c1, v1] = sorted_vertex.top();
      sorted_vertex.pop();
      auto [c2, v2] = sorted_vertex.top();
      sorted_vertex.pop();
      for (auto x : v1) add_bit(x, 0);
      for (auto x : v2) add_bit(x, 1);
      v1.insert(v1.end(), v2.begin(), v2.end());
      c1 += c2;
      sorted_vertex.push({c1, v1});
    }
  }
private:
  std::map<int, std::pair<uint64_t, int>> code_len;
  std::map<std::pair<uint64_t, int>, int> code_index;
  void build_index() {
    for (auto [value, c_l] : code_len) {
      auto [code, len] = c_l;
      code_index[{code, len}] = value;
    }
  }
  void add_bit(int value, int bit) {
    auto& [code, len] = code_len[value];
    code = (code << 1) | bit;
    ++len;
    assert(len <= 64);
  }
};


signed main() {
    std::ignore = freopen("res/raw_huffman_data.txt", "r", stdin);
    std::ignore = freopen("res/prepared_huffman_data.txt", "w", stdout);
    map<string, map<int, ll>> byte_cnt;
    string name;
    int value, cnt;
    while (cin >> name >> value >> cnt) {
        byte_cnt[name][value] += cnt;
    }
    cout << "{\n";
    for (auto &[name, data] : byte_cnt) {
      if (name != "special_cell_type") {
        if (name.substr(0, 8) == "ref_perm") {
          int val = data.begin()->f;
          int len = 0;
          while (val > 0) {
            val /= 10;
            ++len;
          }
          string s(len, '#');
          iota(all(s), '1');
          while (true) {
            int v = stoll(s);
            if (!data.count(v)) data[v] = 0;
            if (!next_permutation(all(s))) break;
          }
        } else {
          for (int i = 0; i < 256; ++i) data[i];
        }
      }
      vector<pair<ll, int>> st_data;
      for (auto [value, cnt] : data) {
        st_data.push_back({cnt, value});
      }
      HuffmanEncoder full_encoder(st_data, "full_" + name);
      rsort(st_data);
      if (sz(st_data) > 256) st_data.resize(256);
      HuffmanEncoder cut_encoder(st_data, "cut_" + name);
      auto x = make_pair(name, st_data);
      cout << pdbg(x) << ",\n";
    }
    cout << "};\n";
    return 0;
}
