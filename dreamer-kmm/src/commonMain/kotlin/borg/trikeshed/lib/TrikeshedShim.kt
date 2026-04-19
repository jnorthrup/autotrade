@file:Suppress("NonAsciiCharacters", "FunctionName", "ObjectPropertyName", "OVERRIDE_BY_INLINE", "UNCHECKED_CAST")

package borg.trikeshed.lib

/**
 * Minimal TrikeShed Join shim for dreamer-kmm standalone compilation.
 * Identical interface to TrikeShed's Join — swapped out via composite build when available.
 */
interface Join<A, B> {
    val a: A
    val b: B
    operator fun component1(): A = a
    operator fun component2(): B = b
}

typealias Twin<T> = Join<T, T>
typealias MetaSeries<I, T> = Join<I, (I) -> T>
typealias Series<T> = MetaSeries<Int, T>

val <T> Series<T>.size: Int get() = a

inline infix fun <A, B> A.j(b: B): Join<A, B> = object : Join<A, B> {
    override val a: A get() = this@j
    override val b: B get() = b
}

inline infix fun <X, C, V : Series<X>> V.α(crossinline xform: (X) -> C): Series<C> =
    size j { i -> xform(this[i]) }

inline operator fun <T> Series<T>.get(i: Int): T = b(i)

fun <T> Series<T>.toList(): List<T> = List(size) { this[it] }

fun <T> Array<T>.toSeries(): Series<T> = size j ::get
fun <T> List<T>.toSeries(): Series<T> = size j ::get
fun DoubleArray.toSeries(): Series<Double> = size j ::get
fun IntArray.toSeries(): Series<Int> = size j ::get
fun CharArray.toSeries(): Series<Char> = size j ::get
fun String.toSeries(): Series<Char> = length j ::get

operator fun <T> Series<T>.iterator(): Iterator<T> = object : Iterator<T> {
    private var current = 0
    override fun hasNext(): Boolean = current < size
    override fun next(): T {
        if (!hasNext()) throw NoSuchElementException()
        return this@iterator[current++]
    }
}

fun <T> Series<T>.isEmpty(): Boolean = a == 0
fun <T> Series<T>.first(): T = this[0]
fun <T> Series<T>.last(): T = this[size - 1]
fun <T> Series<T>.drop(n: Int): Series<T> = (size - n.coerceAtMost(size)) j { i -> this[i + n.coerceAtMost(size)] }
fun <T> Series<T>.take(n: Int): Series<T> = n.coerceAtMost(size) j { this[it] }
fun <T> Series<T>.getOrNull(i: Int): T? = if (i < size) this[i] else null
