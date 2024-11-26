#include <iostream>
using namespace std;

int main(){
    int a, b;
    while(cin >> a >> b){
        int c = a + b;
        int d = 0;
        while(c > 0){
            c /= 10;
            d++;
        }
        cout << d << endl;
    }
    return 0;
}
