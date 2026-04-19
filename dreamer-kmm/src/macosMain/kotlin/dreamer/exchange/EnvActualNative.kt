package dreamer.exchange

import kotlinx.cinterop.toKString
import platform.posix.getenv

actual fun getenv(name: String): String? {
    val ptr = getenv(name)
    return if (ptr != null) ptr.toKString() else null
}
