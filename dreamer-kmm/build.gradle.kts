import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform") version "2.4.0-Beta1"
}

group = "com.vsiwest"
version = "1.5.0-SNAPSHOT"

kotlin {
    @OptIn(ExperimentalKotlinGradlePluginApi::class)
    compilerOptions {
        apiVersion.set(org.jetbrains.kotlin.gradle.dsl.KotlinVersion.KOTLIN_2_4)
        languageVersion.set(org.jetbrains.kotlin.gradle.dsl.KotlinVersion.KOTLIN_2_4)
        freeCompilerArgs = listOf(
            "-opt-in=kotlin.RequiresOptIn",
            "-Xsuppress-version-warnings",
        )
    }

    jvmToolchain(21)

    jvm()

    js(IR) {
        nodejs()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    val hostOs = System.getProperty("os.name")
    if (hostOs == "Mac OS X" && System.getProperty("os.arch") == "aarch64") {
        macosArm64("macos")
    } else if (hostOs == "Linux") {
        linuxX64("linux")
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                // Standalone: TrikeShed abstractions inlined for compilation independence.
                // When TrikeShed's CursorTensor build issues are resolved, switch to:
                //   implementation("org.bereft:trikeshed:1.0")
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
                api("org.jetbrains.kotlinx:kotlinx-datetime:0.7.1-0.6.x-compat")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit"))
            }
        }
        val jsMain by getting
        val jsTest by getting { dependsOn(commonTest) }
        val wasmJsMain by getting
        val wasmJsTest by getting { dependsOn(commonTest) }
    }
}
