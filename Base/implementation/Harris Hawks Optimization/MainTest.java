public class MainTest {
    public static void main(String[] args) {
        double beta = 1.5;
        double sigma = Math.pow((gamma(1 + beta) * Math.sin(Math.PI * beta / 2) / (gamma((1 + beta) / 2) * beta * Math.pow(2, ((beta - 1) / 2)))), (1 / beta));
        System.out.println(sigma);
    }

    static double logGamma(double x) {
        double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
        double ser = 1.0 + 76.18009173 / (x + 0) - 86.50532033 / (x + 1)
                + 24.01409822 / (x + 2) - 1.231739516 / (x + 3)
                + 0.00120858003 / (x + 4) - 0.00000536382 / (x + 5);
        return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
    }

    static double gamma(double x) {
        return Math.exp(logGamma(x));
    }
}
