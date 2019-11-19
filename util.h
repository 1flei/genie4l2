#pragma once

#include <map>
#include <string>
#include <chrono>
#include <iostream>
#include <stack>

struct Result
{
    float key_;
    int id_;
};

const static int MAXK = 100;

template<class ScalarType>
ScalarType sqr(ScalarType x)
{
	return x*x;
}

//FProd :: ScalarType -> ScalarType -> ScalarType
template<class ScalarType, class FProd, class FSum> 
inline ScalarType fast_reduce(int dim, const ScalarType* x, const ScalarType* y, const FProd& fp, const FSum& fs)
{
    unsigned d = dim & ~unsigned(7);
    const ScalarType *aa = x, *end_a = aa + d;
    const ScalarType *bb = y, *end_b = bb + d;
#ifdef __GNUC__
    __builtin_prefetch(aa, 0, 3);
    __builtin_prefetch(bb, 0, 0);
#endif
    ScalarType r = 0.0;
    ScalarType r0, r1, r2, r3, r4, r5, r6, r7;

    const ScalarType *a = end_a, *b = end_b;

    r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0;

    switch (dim & 7) {
        case 7: r6 = fp(a[6], b[6]);
  		// fall through
        case 6: r5 = fp(a[5], b[5]);
  		// fall through
        case 5: r4 = fp(a[4], b[4]);
  		// fall through
        case 4: r3 = fp(a[3], b[3]);
  		// fall through
        case 3: r2 = fp(a[2], b[2]);
  		// fall through
        case 2: r1 = fp(a[1], b[1]);
  		// fall through
        case 1: r0 = fp(a[0], b[0]);
    }

    a = aa; b = bb;
	const auto fsum8 = [&](){
		auto r01 = fs(r0, r1);
		auto r23 = fs(r2, r3);
		auto r45 = fs(r4, r5);
		auto r67 = fs(r6, r7);
		auto r0123 = fs(r01, r23);
		auto r4567 = fs(r45, r67);
		return fs(r0123, r4567);
	};

    for (; a < end_a; a += 8, b += 8) {
#ifdef __GNUC__
        __builtin_prefetch(a + 32, 0, 3);
        __builtin_prefetch(b + 32, 0, 0);
#endif
		r = fs(r, fsum8() );
		r0 = fp(a[0], b[0]);
		r1 = fp(a[1], b[1]);
		r2 = fp(a[2], b[2]);
		r3 = fp(a[3], b[3]);
		r4 = fp(a[4], b[4]);
		r5 = fp(a[5], b[5]);
		r6 = fp(a[6], b[6]);
		r7 = fp(a[7], b[7]);
    }

    r = fs(r, fsum8() );
    return r;
}


template<class ScalarType> 
inline ScalarType calc_l2_sqr(int dim, const ScalarType* x, const ScalarType* y)
{
	const auto fProd = [](ScalarType a, ScalarType b){
		return sqr(a-b);
	};
	const auto fSum = [](ScalarType a, ScalarType b){
		return a+b;
	};
	return fast_reduce(dim, x, y, fProd, fSum);
}


template<class ScalarType> 
inline ScalarType calc_l1_dist(int dim, const ScalarType* x, const ScalarType* y)
{
	const auto fProd = [](ScalarType a, ScalarType b){
		return abs(a-b);
	};
	const auto fSum = [](ScalarType a, ScalarType b){
		return a+b;
	};
	return fast_reduce(dim, x, y, fProd, fSum);
}

template<class ScalarType> 
inline ScalarType calc_inner_product(int dim, const ScalarType* x, const ScalarType* y)
{
	const auto fProd = [](ScalarType a, ScalarType b){
		return a*b;
	};
	const auto fSum = [](ScalarType a, ScalarType b){
		return a+b;
	};
	return fast_reduce(dim, x, y, fProd, fSum);
}

// -----------------------------------------------------------------------------
template<class ScalarType>
ScalarType calc_l2_dist(					// calc L2 distance
	int   dim,							// dimension
	const ScalarType *p1,					// 1st point
	const ScalarType *p2)					// 2nd point
{
	return sqrt(calc_l2_sqr(dim, p1, p2));
}


inline double calc_recall(
    std::vector<double> &res,
    std::vector<double> &ground_truth,
    const double eps = 1e-6)
{
    std::sort(res.begin(), res.end());
    std::sort(ground_truth.begin(), ground_truth.end());

    int n = ground_truth.size();
    int res_idx = std::min(res.size(), ground_truth.size());

    auto it = std::upper_bound(res.begin(), res.begin()+res_idx, ground_truth.back()+eps);
    return std::distance(res.begin(), it) * 1. / n;
}




class MyTimer
{
public:
	typedef std::chrono::duration<double> t_type;
	MyTimer(const std::string &name) :name(name), isRunning(true){
		start();
	};
//	MyTimer(const char *name) :name(name), isRunning(true){
//		start();
//	};
	~MyTimer() {
		end();
	};
	void start() {
		startTime = std::chrono::system_clock::now();
		isRunning = true;
	}
	void end() {
		if(isRunning){
			endTime = std::chrono::system_clock::now();
			t_type t = endTime - startTime;
			// printf(" %s: %f\n", name.c_str(), t.count());
			if(name != ""){
				_tmMap()[name] += t;
				_tmMapCnt()[name]++;
			}
			isRunning = false;
		}
	}
	double getTime(){
		return (endTime-startTime).count();
	} 

	template<typename F, typename... Args>
	static t_type funcTime(F&& func, Args&&... args){
		auto t0 = std::chrono::system_clock::now();
		func(std::forward<Args>(args)...);
		auto t1 = std::chrono::system_clock::now();
		return t_type(t1-t0);
	}

	static void pusht() {
		auto t = std::chrono::system_clock::now();
		return get_tstack().push(t);
	}
	static double popt() {
		auto t = get_tstack().top();
		get_tstack().pop();
        auto duration = std::chrono::duration_cast<t_type>(std::chrono::system_clock::now() - t);
		return duration.count();
	}

	static void clear() {
		_tmMap().clear();
		_tmMapCnt().clear();
	}
	static void clear(const std::string &name) {
		_tmMap()[name] = t_type::zero();
	}
	static double get(const std::string &name) {
		return _tmMap()[name].count();
	}

	static std::stack< std::chrono::time_point<std::chrono::system_clock> >& get_tstack()
	{
		static std::stack< std::chrono::time_point<std::chrono::system_clock> > _tstack;
		return _tstack;
	}
	

//	static double get(const char *name) {
//		//std::string n(name);
//		return g_tmMap[std::string(name)].count();
//	}
//	static int cnt(const char *name) {
//		return g_tmMapCnt[std::string(name)];
//	}
	static int cnt(const std::string &name) {
		return _tmMapCnt()[std::string(name)];
	}
//	static void print(const char *name, std::ostream &outp = std::cout) {
//		outp << "Time for " << name << ": " << get(name) << "  avg:" << get(name) / cnt(name) << std::endl;
//	}
	static void print(const std::string &name, std::ostream &outp = std::cout) {
		outp << "Time for " << name << ": " << get(name) << "  avg:" << get(name) / cnt(name) << "  cnt:" << cnt(name) << std::endl;
	}
	static void printAll(std::ostream &outp = std::cout) {
		for (auto it = _tmMap().begin(); it != _tmMap().end(); it++) {
			print(it->first, outp);
		}
		clear();
	}
	static void printCurTime(std::ostream &outp = std::cout){
		auto time_point = std::chrono::system_clock::now();
	    std::time_t ttp = std::chrono::system_clock::to_time_t(time_point);
		outp << ttp << std::endl;
	}

    template<typename F, typename ...Args>
    static double measure(F func, Args&&... args) {
        auto start = std::chrono::system_clock::now();

        func(std::forward<Args>(args)...);

        auto duration = std::chrono::duration_cast<t_type>(std::chrono::system_clock::now() - start);

        return duration.count();
    }

private:
	static std::map<std::string, t_type> &_tmMap(){
		static std::map<std::string, t_type> g_tmMap;
		return g_tmMap;
	}
	static std::map<std::string, int> &_tmMapCnt(){
		static std::map<std::string, int> g_tmMapCnt;
		return g_tmMapCnt;
	}
//	static std::map<std::string, t_type> g_tmMap;
//	static std::map<std::string, int> g_tmMapCnt;
	std::chrono::time_point<std::chrono::system_clock> startTime;
	std::chrono::time_point<std::chrono::system_clock> endTime;
	std::string name;
	bool isRunning;

};
