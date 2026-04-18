pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
    plugins {
        id("org.jetbrains.kotlin.multiplatform") version "2.4.0-Beta1"
    }
}

rootProject.name = "dreamer-json-support"

includeBuild("../mp/money/TrikeShed")
