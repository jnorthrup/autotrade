import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    kotlin("multiplatform") version "2.4.0-Beta1"
    `maven-publish`
}

group = "org.bereft.dreamer"
version = "1.999"

repositories {
    mavenCentral()
    mavenLocal()
    gradlePluginPortal()
    google()
}

kotlin {
    @OptIn(ExperimentalKotlinGradlePluginApi::class)
    compilerOptions {
        apiVersion.set(org.jetbrains.kotlin.gradle.dsl.KotlinVersion.KOTLIN_2_4)
        languageVersion.set(org.jetbrains.kotlin.gradle.dsl.KotlinVersion.KOTLIN_2_4)
        freeCompilerArgs = listOf(
            "-opt-in=kotlin.RequiresOptIn",
            "-Xsuppress-version-warnings",
            "-Xexpect-actual-classes",
        )
    }

    jvmToolchain(21)

    jvm()

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        binaries.executable()
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("org.bereft:TrikeShed:1.0")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit"))
                implementation("org.junit.jupiter:junit-jupiter:5.9.0")
            }
        }
    }
}

val stageStandaloneDreamerJsonTool by tasks.registering(Sync::class) {
    description = "Stages a standalone Dreamer JSON Wasm tool statically linked against TrikeShed."
    group = "distribution"
    dependsOn("compileDevelopmentExecutableKotlinWasmJs", "wasmJsPackageJson")
    from(layout.buildDirectory.dir("wasm/packages/${project.name}")) {
        include("package.json")
        filter { line: String ->
            line.replace(Regex("\"version\":\\s*\"[^\"]+\""), "\"version\": \"${project.version}\"")
        }
    }
    from(layout.buildDirectory.dir("compileSync/wasmJs/main/developmentExecutable/kotlin")) {
        into("kotlin")
    }
    from(layout.buildDirectory.dir("wasm/node_modules/@js-joda/core")) {
        into("node_modules/@js-joda/core")
    }
    from(layout.projectDirectory.dir("tools")) {
        include("dreamer-json-tool.mjs", "README.md")
    }
    into(layout.buildDirectory.dir("standalone/dreamer-json-tool"))
}

val stageDreamer1999Presentation by tasks.registering(Sync::class) {
    description = "Stages the Dreamer 1.999 presentation bundle with TrikeShed-backed Wasm outputs."
    group = "distribution"
    dependsOn(stageStandaloneDreamerJsonTool)
    from(layout.projectDirectory.file("Dreamer 1.2.js")) {
        rename { "Dreamer 1.999.js" }
    }
    from(layout.buildDirectory.dir("standalone/dreamer-json-tool")) {
        into("wasm")
    }
    into(layout.buildDirectory.dir("presentation/Dreamer-1.999"))
}

tasks.named("assemble") {
    dependsOn(stageStandaloneDreamerJsonTool, stageDreamer1999Presentation)
}
